use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;
use anyhow::{Result, Context};

use crate::profile;
use crate::gate::{AdaptiveGate, StageDecision};
use crate::lag_cache::LagCache;
use crate::resonance;
use crate::grammar;
use crate::prediction;
use crate::context_pack;
use crate::container::{self, IrisContainer,
    FLAG_RESONANCE, FLAG_GRAMMAR, FLAG_PRED_GRAPH, FLAG_CONTEXT_PACK};

// ── Routing thresholds (kept as named constants so they show up in logs) ──
const ZSTD_ROUTE_ENTROPY:      f64 = 0.80;
const ULTRA_COMPRESS_ENTROPY:  f64 = 0.15;
const MAX_RESONANCE_PASSES:    usize = 3;

/// IrisContainer framing overhead: magic + version + lengths + CRC32.
/// Used for the passthrough-guard comparisons.
const CONTAINER_FRAMING: usize = 46;

/// Environment variable that selects a pre-baked `AdaptiveGate` profile.
/// Accepts `default` (production) or `genomic` (the research profile that
/// is biased toward firing structural stages on medium-entropy scientific
/// data). Unknown values fall back to `default`.
pub const IRIS_GATE_ENV: &str = "IRIS_GATE";

fn gate_from_env() -> AdaptiveGate {
    match std::env::var(IRIS_GATE_ENV).as_deref() {
        Ok("genomic") | Ok("research") => AdaptiveGate::genomic_research(),
        _ => AdaptiveGate::default(),
    }
}

// ── Lightweight stage instrumentation ────────────────────────────────────────

struct StageReport {
    name:        &'static str,
    fired:       bool,
    elapsed_ms:  f64,
    bytes_in:    usize,
    bytes_out:   usize,
    entropy_in:  f64,
    entropy_out: f64,
}

impl StageReport {
    fn log(&self) {
        if !self.fired { return; }
        let ratio = if self.bytes_out > 0 {
            self.bytes_in as f64 / self.bytes_out as f64
        } else { 1.0 };
        eprintln!(
            "[iris][stage] {:<10} fired={} {:>8}B→{:>8}B ({:>5.2}x) H={:.3}→{:.3} {:>6.1}ms",
            self.name, self.fired as u8,
            self.bytes_in, self.bytes_out, ratio,
            self.entropy_in, self.entropy_out, self.elapsed_ms
        );
    }
}

pub fn compress(input: &Path, output: &Path) -> Result<()> {
    let t0 = Instant::now();
    let data = std::fs::read(input)
        .with_context(|| format!("cannot read {:?}", input))?;
    let original_len = data.len();
    eprintln!("[iris] input: {}B", original_len);
    if original_len == 0 { anyhow::bail!("empty input"); }
    let original_byte0 = data[0];

    // ── Column grammar fast-path (before profiling) ───────────────────────
    //
    // `infer_columns` gates on "does column grammar produce fewer bytes
    // than the original?"; we still double-check the serialized output
    // including container framing here, so a degenerate column spec can
    // never make the compressed file bigger than the raw input. If it
    // doesn't meaningfully beat raw, we fall through to the profile
    // path where resonance / prediction / entropy coders can still
    // produce a useful result.
    if let Some(cg) = grammar::infer_columns(&data) {
        let grammar_bytes = grammar::serialize_columns(&cg);
        if grammar_bytes.len() + CONTAINER_FRAMING < original_len {
            let col_sz: usize = cg.specs.iter()
                .map(|s| s.data.len()+s.aux.len()+s.aux2.len()).sum();
            eprintln!("[iris] column grammar: delim='{}' cols={} cov={:.0}% → ~{}B ({:.2}x)",
                cg.delimiter as char, cg.num_cols, cg.coverage*100.0, col_sz,
                original_len as f64 / col_sz as f64);
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
        eprintln!("[iris] column grammar produced {}B ≥ raw {}B → falling through to profile path",
            grammar_bytes.len(), original_len);
    }

    // ── Stage 0: Profile (sampling-based — bounded ≤ 64 KB) ──────────────
    let t_prof = Instant::now();
    let prof = profile::profile_fast(&data);
    let prof_ms = t_prof.elapsed().as_secs_f64() * 1000.0;

    let gate_cfg = gate_from_env();
    let decision: StageDecision = gate_cfg.decide(&prof);
    eprintln!(
        "[iris] profile ({:.1}ms sample={}B): H={:.3} Hσ={:.3} skew={:+.2} stab={:.2} D={:.3} acc={:.3}",
        prof_ms, prof.sample_bytes,
        prof.global_entropy, prof.local_entropy_stddev, prof.byte_skewness,
        prof.autocorr_peak_stability, prof.global_disorder, prof.prediction_accuracy,
    );
    eprintln!("[iris] gate: R={} G={} P={} C={} | {}",
        decision.resonance as u8, decision.grammar as u8,
        decision.prediction_graph as u8, decision.context_pack as u8,
        decision.rationale);
    if !prof.resonance_lags.is_empty() {
        eprintln!("[iris] resonance lags: {:?}",
            prof.resonance_lags.iter().take(4)
                .map(|r| format!("{}:{:.3}", r.lag, r.strength)).collect::<Vec<_>>());
    }

    if decision.passthrough {
        return write_passthrough(&data, output);
    }

    let mut working = data.clone();
    let mut flags   = 0u8;
    let mut resonance_hdr_bytes = vec![];
    let mut grammar_bytes       = vec![];
    let mut pred_meta_bytes     = vec![];

    // ── Stage 1: Multi-pass Resonance (with LagCache) ────────────────────
    let mut resonance_stack: Vec<resonance::ResonanceHeader> = vec![];
    let mut lag_cache = LagCache::new();
    if decision.resonance {
        for pass in 0..MAX_RESONANCE_PASSES {
            let t_pass = Instant::now();
            let before_h = profile::byte_entropy(&working);

            // Lag candidates for this pass:
            //   pass 0: take them from the initial profile
            //   pass n: first try the LagCache (probe sample only),
            //   else: run the cheap sampled profile on the residual.
            let pass_lags: Vec<profile::ResonanceLag> = if pass == 0 {
                prof.resonance_lags.clone()
            } else if let Some(hit) = lag_cache.probe(&working) {
                vec![hit]
            } else {
                profile::profile_fast(&working).resonance_lags
            };

            if pass_lags.is_empty() { break; }

            match resonance::extract(&working, &pass_lags) {
                Some((hdr, residual)) => {
                    let after_h = profile::byte_entropy(&residual);
                    lag_cache.note(hdr.lag as usize, hdr.strength as f64);
                    eprintln!(
                        "[iris] resonance pass {} lag={} s={:.3} {}→{}B (H {:.3}→{:.3}) {:.1}ms",
                        pass+1, hdr.lag, hdr.strength, working.len(), residual.len(),
                        before_h, after_h, t_pass.elapsed().as_secs_f64()*1000.0
                    );
                    resonance_stack.push(hdr);
                    working = residual;
                    flags |= FLAG_RESONANCE;
                    if after_h < ULTRA_COMPRESS_ENTROPY { break; }
                }
                None => break,
            }
        }
        if lag_cache.hits + lag_cache.misses > 0 {
            eprintln!("[iris] lag_cache: {} hit / {} miss",
                lag_cache.hits, lag_cache.misses);
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
    if decision.grammar && !prof.resonance_lags.is_empty() {
        let t_g = Instant::now();
        let bytes_in = working.len();
        let h_in = profile::byte_entropy(&working);
        let stride = prof.resonance_lags[0].lag;
        if let Some(tmpl) = grammar::infer(&working, stride) {
            if !tmpl.field_offsets.is_empty() && tmpl.coverage > grammar::GRAMMAR_MIN_COVERAGE {
                let field_flat: Vec<u8> = tmpl.field_values.iter()
                    .flat_map(|fv| fv.iter().cloned()).collect();
                if !field_flat.is_empty() {
                    grammar_bytes = grammar::serialize(&tmpl);
                    let bytes_out = field_flat.len();
                    let h_out = profile::byte_entropy(&field_flat);
                    working = field_flat;
                    flags |= FLAG_GRAMMAR;
                    StageReport {
                        name: "grammar", fired: true,
                        elapsed_ms: t_g.elapsed().as_secs_f64()*1000.0,
                        bytes_in, bytes_out, entropy_in: h_in, entropy_out: h_out,
                    }.log();
                    eprintln!("[iris] grammar stride={} cov={:.1}%",
                        tmpl.stride, tmpl.coverage*100.0);
                }
            }
        }
    }

    let working_entropy = profile::byte_entropy(&working);
    eprintln!("[iris] working: {}B entropy={:.3}", working.len(), working_entropy);

    // ── Stage 3: Prediction Graph ────────────────────────────────────────
    if decision.prediction_graph
        && working_entropy >= ULTRA_COMPRESS_ENTROPY
        && working.len() >= prediction::PRED_BLOCK_SIZE * 2
    {
        let t_p = Instant::now();
        let bytes_in = working.len();
        let fps = compute_fingerprints(&working, prediction::PRED_BLOCK_SIZE);
        if fps.len() > 1 {
            let encodings = prediction::build(&working, &fps);
            let dc = encodings.iter()
                .filter(|e| matches!(e, prediction::BlockEncoding::Delta{..})).count();
            eprintln!("[iris] pred graph {}/{} delta", dc, encodings.len());
            if dc > 0 {
                let (payload, metas) = prediction::flatten_for_packing(&encodings);
                pred_meta_bytes = container::serialize_pred_meta(&metas);
                let h_out = profile::byte_entropy(&payload);
                let h_in  = profile::byte_entropy(&working);
                let bytes_out = payload.len();
                working = payload;
                flags |= FLAG_PRED_GRAPH;
                StageReport {
                    name: "pred-graph", fired: true,
                    elapsed_ms: t_p.elapsed().as_secs_f64()*1000.0,
                    bytes_in, bytes_out, entropy_in: h_in, entropy_out: h_out,
                }.log();
            }
        }
    }

    let working_byte0 = working.first().copied().unwrap_or(0);
    let working_entropy2 = profile::byte_entropy(&working);

    // ── Stage 4+5: Route ─────────────────────────────────────────────────
    let video_payload; let scatter_bytes; let ctx_hdr_bytes; let encoder_byte: u8;

    if working_entropy2 < ULTRA_COMPRESS_ENTROPY {
        let t_c = Instant::now();
        eprintln!("[iris] ultra-compress: range coder (H={:.3})", working_entropy2);
        let compressed = crate::range_coder::encode(&working);
        eprintln!("[iris] range: {}B → {}B ({:.1}x) {:.1}ms",
            working.len(), compressed.len(),
            working.len() as f64 / compressed.len().max(1) as f64,
            t_c.elapsed().as_secs_f64()*1000.0);
        video_payload  = compressed;
        scatter_bytes  = vec![];
        ctx_hdr_bytes  = vec![];
        encoder_byte   = 0xFC;
        flags          |= FLAG_CONTEXT_PACK;

    } else if working_entropy2 < ZSTD_ROUTE_ENTROPY {
        let t_c = Instant::now();
        eprintln!("[iris] rANS-route");

        // Build the context table only now — this is the only route that
        // actually needs it. On large files this saves us the 4·N bucket
        // memory that the old pipeline always allocated upfront.
        let ctx_table = profile::build_context_table(&working);
        let (flat, position_lists, hdr) =
            context_pack::pack_with_positions(&working, &ctx_table);
        eprintln!("[iris] ctx pack {} contexts (bucket mem ≈ {}B)",
            hdr.context_map.len(), ctx_table.memory_bytes());

        let scatter = container::serialize_scatter(&position_lists)?;
        let ctx_h   = context_pack::serialize_header(&hdr);
        let compressed = crate::rans::encode(&flat);
        eprintln!("[iris] rANS flat: {}B → {}B ({:.1}x) {:.1}ms",
            flat.len(), compressed.len(),
            flat.len() as f64/compressed.len().max(1) as f64,
            t_c.elapsed().as_secs_f64()*1000.0);

        video_payload  = compressed;
        scatter_bytes  = scatter;
        ctx_hdr_bytes  = ctx_h;
        encoder_byte   = 0xFF;
        flags          |= FLAG_CONTEXT_PACK;

    } else {
        let t_c = Instant::now();
        eprintln!("[iris] block-match route");
        let compressed = crate::block_match::compress(&working);
        eprintln!("[iris] block-match: {}B → {}B ({:.1}x) {:.1}ms",
            working.len(), compressed.len(),
            working.len() as f64 / compressed.len().max(1) as f64,
            t_c.elapsed().as_secs_f64()*1000.0);

        video_payload  = compressed;
        scatter_bytes  = vec![];
        ctx_hdr_bytes  = vec![];
        encoder_byte   = 0xFD;
        flags          |= FLAG_CONTEXT_PACK;
    }

    // ── Passthrough guard: if total output would be larger than input,
    //    fall back to raw passthrough. This is the "never make files
    //    bigger" promise. CONTAINER_FRAMING accounts for the fixed
    //    magic+version+lengths+CRC32 bytes the outer container adds.
    let total_meta: usize = resonance_hdr_bytes.len()
        + grammar_bytes.len() + pred_meta_bytes.len()
        + scatter_bytes.len() + ctx_hdr_bytes.len() + video_payload.len()
        + CONTAINER_FRAMING;
    if total_meta >= original_len && original_len > 0 {
        eprintln!("[iris] structural stages did not beat raw ({}B) → passthrough guard",
            total_meta);
        return write_passthrough(&data, output);
    }

    let c = IrisContainer {
        encoder: encoder_byte,
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
        let dec = if c.encoder == 0xFB {
            c.video.clone()
        } else {
            crate::rans::decode(&c.video)
                .ok_or_else(|| anyhow::anyhow!("corrupt passthrough data"))?
        };
        std::fs::write(output, &dec[..original_len.min(dec.len())])?;
        return Ok(());
    }

    // ── Decompress video/flat ─────────────────────────────────────────────
    let working = if c.has_context_pack() {
        if c.encoder == 0xFC {
            crate::range_coder::decode(&c.video)
                .ok_or_else(|| anyhow::anyhow!("corrupt range-coded data"))?
        } else if c.encoder == 0xFF {
            let (hdr, _) = context_pack::deserialize_header(&c.ctx_hdr)
                .ok_or_else(|| anyhow::anyhow!("corrupt ctx hdr"))?;
            let flat = crate::rans::decode(&c.video)
                .ok_or_else(|| anyhow::anyhow!("corrupt rANS data"))?;
            let position_lists = container::deserialize_scatter(&c.scatter_map)?;
            context_pack::unpack_full(&hdr, &flat, c.byte0, &position_lists)
        } else if c.encoder == 0xFD {
            crate::block_match::decompress(&c.video)
                .ok_or_else(|| anyhow::anyhow!("corrupt block-match data"))?
        } else {
            anyhow::bail!("unsupported encoder byte {:#04x} (legacy AV1?)", c.encoder);
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

/// Emit just the profile + gate decision for a file (no compression).
/// Useful for the genomic-harness experiments.
pub fn profile_only(input: &Path) -> Result<()> {
    let data = std::fs::read(input)
        .with_context(|| format!("cannot read {:?}", input))?;
    if data.is_empty() { anyhow::bail!("empty input"); }
    let prof = profile::profile_fast(&data);
    let gate_cfg = gate_from_env();
    let decision = gate_cfg.decide(&prof);

    println!("file              : {:?}", input);
    println!("bytes             : {}", data.len());
    println!("sample_bytes      : {}", prof.sample_bytes);
    println!("global_entropy    : {:.4}", prof.global_entropy);
    println!("local_entropy_std : {:.4}", prof.local_entropy_stddev);
    println!("byte_mean         : {:.2}", prof.byte_mean);
    println!("byte_variance     : {:.2}", prof.byte_variance);
    println!("byte_skewness     : {:+.4}", prof.byte_skewness);
    println!("pred_accuracy     : {:.4}", prof.prediction_accuracy);
    println!("disorder          : {:.4}", prof.global_disorder);
    println!("peak_stability    : {:.4}", prof.autocorr_peak_stability);
    if let Some(l) = prof.resonance_lags.first() {
        println!("dominant_lag      : {} (strength={:.3})", l.lag, l.strength);
    } else {
        println!("dominant_lag      : (none)");
    }
    println!("gate.resonance    : {}", decision.resonance);
    println!("gate.grammar      : {}", decision.grammar);
    println!("gate.pred_graph   : {}", decision.prediction_graph);
    println!("gate.context_pack : {}", decision.context_pack);
    println!("gate.passthrough  : {}", decision.passthrough);
    println!("gate.rationale    : {}", decision.rationale);
    Ok(())
}

fn write_passthrough(data: &[u8], output: &Path) -> Result<()> {
    let encoded = crate::rans::encode(data);
    let (video_data, encoder_byte) = if encoded.len() >= data.len() {
        (data.to_vec(), 0xFB)
    } else {
        (encoded, 0xFF)
    };

    let c = IrisContainer {
        encoder: encoder_byte, flags: 0, original_len: data.len() as u64,
        byte0: data[0], original_byte0: data[0],
        resonance_hdr: vec![], grammar_data: vec![],
        pred_meta: vec![], scatter_map: vec![], ctx_hdr: vec![],
        video: video_data,
    };
    let f = std::fs::File::create(output)?;
    c.write(&mut BufWriter::new(f))?;
    Ok(())
}

/// Compute SimHash fingerprints for each block.
fn compute_fingerprints(data: &[u8], block_size: usize) -> Vec<u64> {
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut fps = Vec::with_capacity(n_blocks);

    for block_start in (0..data.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(data.len());
        let block = &data[block_start..block_end];
        fps.push(simhash_block(block));
    }
    fps
}

/// SimHash of a single block using 4-byte shingles.
pub fn simhash_block(block: &[u8]) -> u64 {
    let mut weights = [0i32; 64];

    if block.len() < 4 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in block {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        return h;
    }

    for window in block.windows(4) {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in window {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for bit in 0..64 {
            if (h >> bit) & 1 == 1 {
                weights[bit] += 1;
            } else {
                weights[bit] -= 1;
            }
        }
    }

    let mut fp: u64 = 0;
    for bit in 0..64 {
        if weights[bit] > 0 {
            fp |= 1u64 << bit;
        }
    }
    fp
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
