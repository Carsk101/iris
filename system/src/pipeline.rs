use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;
use anyhow::{Result, Context};

use crate::profile::{self, StageGate};
use crate::resonance;
use crate::grammar;
use crate::prediction;
use crate::context_pack;
use crate::columnar;
use crate::encoder::{self, Encoder};
use crate::container::{self, IrisContainer,
    FLAG_RESONANCE, FLAG_GRAMMAR, FLAG_PRED_GRAPH, FLAG_CONTEXT_PACK, FLAG_COLUMNAR};

const ZSTD_ROUTE_ENTROPY:      f64 = 0.80;
const ULTRA_COMPRESS_ENTROPY:  f64 = 0.15;
const MAX_RESONANCE_PASSES:    usize = 3;

pub fn compress(input: &Path, output: &Path) -> Result<()> {
    let t0 = Instant::now();
    let data = std::fs::read(input)
        .with_context(|| format!("cannot read {:?}", input))?;
    let original_len = data.len();
    eprintln!("[iris] input: {}B", original_len);
    if original_len == 0 { anyhow::bail!("empty input"); }
    let original_byte0 = data[0];

    // ── Column grammar fast-path (before profiling) ───────────────────────
    if let Some(cg) = grammar::infer_columns(&data) {
        let col_sz: usize = cg.specs.iter().map(|s| s.data.len()+s.aux.len()+s.aux2.len()).sum();
        eprintln!("[iris] column grammar: delim='{}' cols={} cov={:.0}% → ~{}B ({:.2}x)",
            cg.delimiter as char, cg.num_cols, cg.coverage*100.0, col_sz,
            original_len as f64 / col_sz as f64);
        let grammar_bytes = grammar::serialize_columns(&cg);
        let c = IrisContainer {
            encoder: 0xFE, flags: FLAG_GRAMMAR,
            original_len: original_len as u64,
            byte0: original_byte0, original_byte0,
            resonance_hdr: vec![], grammar_data: grammar_bytes,
            pred_meta: vec![], scatter_map: vec![], ctx_hdr: vec![], video: vec![],
        };
        let f = std::fs::File::create(output)?;
        c.write(&mut BufWriter::new(f))?;
        let sz = std::fs::metadata(output)?.len() as usize;
        eprintln!("[iris] done {:.1}ms | {}→{}B | {:.3}x | flags={:04b}",
            t0.elapsed().as_secs_f64()*1000.0,
            original_len, sz, original_len as f64/sz as f64, FLAG_GRAMMAR);
        return Ok(());
    }

    // ── Stage 0: Profile ──────────────────────────────────────────────────
    let prof = profile::profile(&data);
    let gate = StageGate::from_profile(&prof);
    eprintln!("[iris] H={:.3} D={:.3} acc={:.3} | R={} G={} P={} C={}",
        prof.global_entropy, prof.global_disorder, prof.prediction_accuracy,
        gate.resonance as u8, gate.grammar as u8,
        gate.prediction_graph as u8, gate.context_pack as u8);
    if !prof.resonance_lags.is_empty() {
        eprintln!("[iris] resonance lags: {:?}",
            prof.resonance_lags.iter().take(4)
                .map(|r| format!("{}:{:.3}", r.lag, r.strength)).collect::<Vec<_>>());
    }

    if gate.passthrough {
        return write_passthrough(&data, output);
    }

    let mut working = data.clone();
    let mut flags   = 0u8;
    let mut resonance_hdr_bytes = vec![];
    let mut grammar_bytes       = vec![];
    let mut pred_meta_bytes     = vec![];

    // ── Stage 1: Multi-pass Resonance ────────────────────────────────────
    let mut resonance_stack: Vec<resonance::ResonanceHeader> = vec![];
    if gate.resonance {
        for pass in 0..MAX_RESONANCE_PASSES {
            let pass_prof = if pass == 0 { prof.resonance_lags.clone() } else {
                profile::profile(&working).resonance_lags
            };
            if pass_prof.is_empty() { break; }
            match resonance::extract(&working, &pass_prof) {
                Some((hdr, residual)) => {
                    eprintln!("[iris] resonance pass {} lag={} s={:.3} {}→{}B",
                        pass+1, hdr.lag, hdr.strength, working.len(), residual.len());
                    resonance_stack.push(hdr);
                    working = residual;
                    flags |= FLAG_RESONANCE;
                    // Stop if residual is near-trivial (ultra-compress will take it)
                    if profile::byte_entropy(&working) < ULTRA_COMPRESS_ENTROPY { break; }
                }
                None => break,
            }
        }
    }
    // Serialize resonance stack: [n_passes: u8][hdr0][hdr1]...
    if !resonance_stack.is_empty() {
        resonance_hdr_bytes.push(resonance_stack.len() as u8);
        for hdr in &resonance_stack {
            resonance_hdr_bytes.extend_from_slice(&resonance::serialize_header(hdr));
        }
    }

    // ── Stage 2: Grammar ─────────────────────────────────────────────────
    if gate.grammar && !prof.resonance_lags.is_empty() {
        let stride = prof.resonance_lags[0].lag;
        if let Some(tmpl) = grammar::infer(&working, stride) {
            if !tmpl.field_offsets.is_empty() && tmpl.coverage > grammar::GRAMMAR_MIN_COVERAGE {
                let field_flat: Vec<u8> = tmpl.field_values.iter()
                    .flat_map(|fv| fv.iter().cloned()).collect();
                if !field_flat.is_empty() {
                    eprintln!("[iris] grammar stride={} cov={:.1}%", tmpl.stride, tmpl.coverage*100.0);
                    grammar_bytes = grammar::serialize(&tmpl);
                    working = field_flat;
                    flags |= FLAG_GRAMMAR;
                }
            }
        }
    }

    let working_entropy = profile::byte_entropy(&working);
    eprintln!("[iris] working: {}B entropy={:.3}", working.len(), working_entropy);

    // ── Stage 3: Prediction Graph — SKIP if ultra-compress will take it ─
    if gate.prediction_graph
        && working_entropy >= ULTRA_COMPRESS_ENTROPY   // only if not ultra
        && working.len() >= prediction::PRED_BLOCK_SIZE * 2
    {
        let fps = compute_fingerprints(&working, prediction::PRED_BLOCK_SIZE);
        if fps.len() > 1 {
            let encodings = prediction::build(&working, &fps);
            let dc = encodings.iter().filter(|e| matches!(e, prediction::BlockEncoding::Delta{..})).count();
            eprintln!("[iris] pred graph {}/{} delta", dc, encodings.len());
            if dc > 0 {
                let (payload, metas) = prediction::flatten_for_packing(&encodings);
                pred_meta_bytes = container::serialize_pred_meta(&metas);
                working = payload;
                flags |= FLAG_PRED_GRAPH;
            }
        }
    }

    let working_byte0 = working.first().copied().unwrap_or(0);
    let working_entropy2 = profile::byte_entropy(&working);

    // ── Stage 4+5: Route ─────────────────────────────────────────────────
    let video_payload; let scatter_bytes; let ctx_hdr_bytes; let is_av1; let enc;

    if working_entropy2 < ULTRA_COMPRESS_ENTROPY {
        eprintln!("[iris] ultra-compress: bzip2 (H={:.3})", working_entropy2);
        use bzip2::Compression;
        use bzip2::read::BzEncoder as BzEnc;
        use std::io::Read;
        let mut bz = BzEnc::new(std::io::Cursor::new(&working), Compression::best());
        let mut compressed = Vec::new();
        bz.read_to_end(&mut compressed)?;
        eprintln!("[iris] bzip2: {}B → {}B ({:.1}x)",
            working.len(), compressed.len(),
            working.len() as f64 / compressed.len() as f64);
        video_payload  = compressed;
        scatter_bytes  = vec![];
        ctx_hdr_bytes  = vec![];
        is_av1         = false;
        enc            = Encoder::LibX264;
        flags          |= FLAG_CONTEXT_PACK;

    } else if working_entropy2 < ZSTD_ROUTE_ENTROPY || encoder::detect() == Encoder::LibX264 {
        eprintln!("[iris] zstd-route");
        let working_prof = profile::profile(&working);
        let (flat, position_lists, hdr) =
            context_pack::pack_with_positions(&working, &working_prof.context_table);
        eprintln!("[iris] ctx pack {} contexts", hdr.context_map.len());

        let scatter = container::serialize_scatter(&position_lists)?;
        let ctx_h   = context_pack::serialize_header(&hdr);
        let compressed = zstd::encode_all(std::io::Cursor::new(&flat), 22)?;
        eprintln!("[iris] zstd flat: {}B → {}B ({:.1}x)",
            flat.len(), compressed.len(), flat.len() as f64/compressed.len() as f64);

        video_payload  = compressed;
        scatter_bytes  = scatter;
        ctx_hdr_bytes  = ctx_h;
        is_av1         = false;
        enc            = Encoder::LibX264;
        flags          |= FLAG_CONTEXT_PACK;

    } else {
        eprintln!("[iris] AV1-route");
        let working_prof  = profile::profile(&working);
        let (hdr, yuv)    = context_pack::pack(&working, &working_prof.context_table);
        let position_lists: Vec<Vec<u32>> = hdr.context_map.iter()
            .map(|e| working_prof.context_table.buckets[e.context_byte as usize].clone())
            .collect();
        let scatter = container::serialize_scatter(&position_lists)?;
        let ctx_h   = context_pack::serialize_header(&hdr);
        let det_enc = encoder::detect();
        let encoded = encoder::encode(&yuv, hdr.frame_width as usize,
            hdr.frame_height as usize, hdr.frame_count as usize, det_enc)?;

        video_payload  = encoded;
        scatter_bytes  = scatter;
        ctx_hdr_bytes  = ctx_h;
        is_av1         = true;
        enc            = det_enc;
        flags          |= FLAG_CONTEXT_PACK;
    }

    let c = IrisContainer {
        encoder: if is_av1 { enc.to_u8() } else { 0xFF },
        flags, original_len: original_len as u64,
        byte0: working_byte0, original_byte0,
        resonance_hdr: resonance_hdr_bytes, grammar_data: grammar_bytes,
        pred_meta: pred_meta_bytes, scatter_map: scatter_bytes,
        ctx_hdr: ctx_hdr_bytes, video: video_payload,
    };

    let f = std::fs::File::create(output)?;
    c.write(&mut BufWriter::new(f))?;
    let out_size = std::fs::metadata(output)?.len() as usize;
    let ratio    = original_len as f64 / out_size as f64;
    eprintln!("[iris] done {:.1}ms | {}→{}B | {:.3}x | {:.1}% | flags={:05b}",
        t0.elapsed().as_secs_f64()*1000.0,
        original_len, out_size, ratio, (1.0-1.0/ratio)*100.0, flags);
    Ok(())
}

pub fn decompress(input: &Path, output: &Path) -> Result<()> {
    let t0 = Instant::now();
    let f  = std::fs::File::open(input)?;
    let c  = IrisContainer::read(&mut BufReader::new(f))?;
    eprintln!("[iris] flags={:05b} original={}B", c.flags, c.original_len);
    let original_len = c.original_len as usize;

    // ── Column grammar path (enc=0xFE) ────────────────────────────────────
    if c.encoder == 0xFE && c.flags & FLAG_GRAMMAR != 0 {
        eprintln!("[iris] column grammar decompress");
        let (cg, _) = grammar::deserialize_columns(&c.grammar_data)
            .ok_or_else(|| anyhow::anyhow!("corrupt column grammar"))?;
        let reconstructed = grammar::reconstruct_columns(&cg, original_len);
        let wl = reconstructed.len().min(original_len);
        std::fs::write(output, &reconstructed[..wl])?;
        eprintln!("[iris] wrote {}B in {:.1}ms", wl, t0.elapsed().as_secs_f64()*1000.0);
        return Ok(());
    }

    // ── Passthrough ───────────────────────────────────────────────────────
    if c.flags == 0 {
        let dec = zstd::decode_all(std::io::Cursor::new(&c.video))?;
        std::fs::write(output, &dec[..original_len.min(dec.len())])?;
        return Ok(());
    }

    // ── Decompress video/flat ─────────────────────────────────────────────
    let working = if c.has_context_pack() {
        if c.ctx_hdr.is_empty() {
            // Ultra-compress: video = bzip2-compressed working data
            use bzip2::read::BzDecoder;
            use std::io::Read;
            let mut bz = BzDecoder::new(std::io::Cursor::new(&c.video));
            let mut dec = Vec::new();
            bz.read_to_end(&mut dec).unwrap_or(0);
            dec
        } else if c.encoder == 0xFF {
            // zstd-route
            let (hdr, _) = context_pack::deserialize_header(&c.ctx_hdr)
                .ok_or_else(|| anyhow::anyhow!("corrupt ctx hdr"))?;
            let flat = zstd::decode_all(std::io::Cursor::new(&c.video))?;
            let position_lists = container::deserialize_scatter(&c.scatter_map)?;
            context_pack::unpack_full(&hdr, &flat, c.byte0, &position_lists)
        } else {
            // AV1-route
            let (hdr, _) = context_pack::deserialize_header(&c.ctx_hdr)
                .ok_or_else(|| anyhow::anyhow!("corrupt ctx hdr"))?;
            let enc = Encoder::from_u8(c.encoder);
            let yuv = encoder::decode(&c.video,
                hdr.frame_width as usize, hdr.frame_height as usize,
                hdr.frame_count as usize, enc)?;
            let position_lists = container::deserialize_scatter(&c.scatter_map)?;
            let flat = extract_y_planes(&yuv, &hdr);
            context_pack::unpack_full(&hdr, &flat, c.byte0, &position_lists)
        }
    } else {
        c.video.clone()
    };

    // ── Stage 3↑: Pred graph ─────────────────────────────────────────────
    let working = if c.has_pred_graph() && !c.pred_meta.is_empty() {
        let metas = container::deserialize_pred_meta(&c.pred_meta)
            .ok_or_else(|| anyhow::anyhow!("corrupt pred meta"))?;
        rebuild_from_pred(&working, &metas)
    } else { working };

    // ── Stage 2↑: Grammar ─────────────────────────────────────────────────
    let working = if c.has_grammar() && !c.grammar_data.is_empty() {
        if let Some((tmpl, _)) = grammar::deserialize(&c.grammar_data) {
            let nf = tmpl.field_offsets.len();
            let ni = if nf > 0 { working.len() / nf } else { 0 };
            let mut nt = tmpl.clone(); nt.field_values.clear();
            for fi in 0..nf {
                let s = fi * ni; let e = (s + ni).min(working.len());
                nt.field_values.push(working[s..e].to_vec());
            }
            grammar::reconstruct(&nt)
        } else { working }
    } else { working };

    // ── Stage 1↑: Multi-pass Resonance ────────────────────────────────────
    let working = if c.has_resonance() && !c.resonance_hdr.is_empty() {
        let mut headers = vec![];
        let n_passes = c.resonance_hdr[0] as usize;
        let mut pos = 1;
        for _ in 0..n_passes {
            if let Some((hdr, consumed)) = resonance::deserialize_header(&c.resonance_hdr[pos..]) {
                headers.push(hdr); pos += consumed;
            } else { break; }
        }
        // Apply in reverse order
        let mut w = working;
        for hdr in headers.iter().rev() {
            w = resonance::reconstruct(hdr, &w);
        }
        w
    } else { working };

    let wl = working.len().min(original_len);
    std::fs::write(output, &working[..wl])?;
    eprintln!("[iris] wrote {}B in {:.1}ms", wl, t0.elapsed().as_secs_f64()*1000.0);
    Ok(())
}

pub fn info(input: &Path) -> Result<()> {
    let f = std::fs::File::open(input)?;
    let c = IrisContainer::read(&mut BufReader::new(f))?;
    let cs = std::fs::metadata(input)?.len();
    let r  = c.original_len as f64 / cs as f64;
    println!("iris v2 — {:?}", input);
    println!("  original  : {}B", c.original_len);
    println!("  compressed: {}B", cs);
    println!("  ratio     : {:.3}x ({:.1}% savings)", r, (1.0-1.0/r)*100.0);
    Ok(())
}

fn write_passthrough(data: &[u8], output: &Path) -> Result<()> {
    let c = IrisContainer {
        encoder: 0xFF, flags: 0, original_len: data.len() as u64,
        byte0: data[0], original_byte0: data[0],
        resonance_hdr: vec![], grammar_data: vec![],
        pred_meta: vec![], scatter_map: vec![], ctx_hdr: vec![],
        video: zstd::encode_all(std::io::Cursor::new(data), 22)?,
    };
    let f = std::fs::File::create(output)?;
    c.write(&mut BufWriter::new(f))?;
    Ok(())
}

fn compute_fingerprints(data: &[u8], block_size: usize) -> Vec<u64> {
    let mut fps = Vec::new();
    let mut hash: u64 = 0xdeadbeef_cafebabe;
    let mut count = 0usize;
    for &b in data {
        hash = hash.wrapping_mul(6364136223846793005).wrapping_add(b as u64 ^ 1442695040888963407);
        count += 1;
        if count >= block_size { fps.push(hash); hash = 0; count = 0; }
    }
    if count > 0 { fps.push(hash); }
    fps
}

fn rebuild_from_pred(working: &[u8], metas: &[crate::prediction::BlockMeta]) -> Vec<u8> {
    if metas.is_empty() { return working.to_vec(); }
    let mut decoded: Vec<Option<Vec<u8>>> = vec![None; metas.len()];
    let mut sorted: Vec<&crate::prediction::BlockMeta> = metas.iter().collect();
    sorted.sort_by_key(|m| m.block_idx);
    for m in &sorted {
        let idx = m.block_idx as usize;
        let off = m.offset as usize; let len = m.len as usize;
        let slice = working.get(off..off+len).unwrap_or(&[]);
        if !m.is_delta {
            decoded[idx] = Some(slice.to_vec());
        } else {
            let ref_data = decoded.get(m.ref_block as usize)
                .and_then(|d| d.as_ref()).map(|v| v.as_slice()).unwrap_or(&[]);
            let out: Vec<u8> = (0..len)
                .map(|i| ref_data.get(i).copied().unwrap_or(0) ^ slice.get(i).copied().unwrap_or(0))
                .collect();
            decoded[idx] = Some(out);
        }
    }
    decoded.into_iter().filter_map(|d| d).flatten().collect()
}

fn extract_y_planes(yuv: &[u8], hdr: &context_pack::ContextPackHeader) -> Vec<u8> {
    let w  = hdr.frame_width  as usize;
    let h  = hdr.frame_height as usize;
    let nf = hdr.frame_count  as usize;
    let pf = w * h;
    let fs = pf * 3 / 2;
    let mut flat = Vec::with_capacity(pf * nf);
    for fi in 0..nf {
        let y0 = fi * fs; let end = (y0 + pf).min(yuv.len());
        flat.extend_from_slice(&yuv[y0..end]);
    }
    flat
}
