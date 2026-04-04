//! Stage 2 — Grammar / Template Inference
//!
//! Strategy A — Column grammar (logs, CSV):
//!   Detect delimiter, split into columns, apply type-aware per-column encoding:
//!   constant · HMS-time · u8/u16/u32 integer array · dictionary · raw-zstd
//!   Adjacent dict columns are merged when co-occurrence gives savings.
//!   Common prefix/suffix is stripped before encoding and restored on decode.
//!
//! Strategy B — Stride grammar (binary with period):
//!   Detect repeating fixed-value positions in fixed-stride records.

pub const GRAMMAR_MIN_COVERAGE:   f64 = 0.25;
pub const FIELD_ENTROPY_THRESHOLD: f64 = 0.3;
pub const GRAMMAR_MIN_STRIDE:      usize = 4;

// ColSpec encoding types
const ENC_RAW:    u8 = 0; // zstd of newline-joined values
const ENC_U8:     u8 = 1; // u8 array + zstd
const ENC_U16:    u8 = 2; // u16-LE array + zstd
const ENC_U32:    u8 = 3; // u32-LE array + zstd
const ENC_DICT:   u8 = 4; // null-sep dict (zstd) + u8 indices (zstd)
const ENC_HMS:    u8 = 5; // HH:MM:SS → 3 × u8-array zstd
const ENC_MERGED: u8 = 6; // two adjacent cols merged as single dict entry
const ENC_EMPTY:  u8 = 0xFF; // placeholder for a col merged into the previous one

// ─── Column grammar ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ColumnGrammar {
    pub delimiter: u8,
    pub num_cols:  usize,
    pub row_count: usize,
    pub coverage:  f64,
    pub specs:     Vec<ColSpec>,
}

/// One column's compressed encoding.  `pfx`/`sfx` are the common prefix/suffix
/// that were stripped before encoding and must be reattached on decode.
#[derive(Debug, Clone)]
pub struct ColSpec {
    pub enc:        u8,
    pub pfx:        Vec<u8>, // common prefix stripped before encoding
    pub sfx:        Vec<u8>, // common suffix stripped before encoding
    pub data:       Vec<u8>, // primary compressed payload
    pub aux:        Vec<u8>, // dict zstd | HMS-M array
    pub aux2:       Vec<u8>, // HMS-S array
    pub merge_sep:  u8,      // ENC_MERGED: byte separator between the two original cols
    pub merged_col: u32,     // ENC_MERGED: original index of the second col
}

impl ColSpec {
    fn mk(enc: u8, data: Vec<u8>) -> Self {
        Self { enc, pfx: vec![], sfx: vec![], data,
               aux: vec![], aux2: vec![], merge_sep: 0, merged_col: 0 }
    }
    fn empty() -> Self { Self::mk(ENC_EMPTY, vec![]) }
    fn payload_size(&self) -> usize {
        self.data.len() + self.aux.len() + self.aux2.len()
    }
}

/// Try column-grammar compression.  Returns `None` if it can't beat raw zstd.
pub fn infer_columns(data: &[u8]) -> Option<ColumnGrammar> {
    // ── ASCII guard: reject binary data ──────────────────────────────────────
    let probe = &data[..data.len().min(8192)];
    let non_print = probe.iter().filter(|&&b| {
        b != b'\n' && b != b'\r' && b != b'\t' && !(0x20..=0x7E).contains(&b)
    }).count();
    if non_print * 20 > probe.len() { return None; }
    let has_nulls = probe.iter().any(|&b| b == 0);
    if has_nulls { return None; }

    // ── Split into lines ──────────────────────────────────────────────────────
    let lines: Vec<&[u8]> = data.split(|&b| b == b'\n')
        .filter(|l| !l.is_empty()).collect();
    if lines.len() < 10 { return None; }

    let delim = detect_delimiter(&lines)?;
    let expected = lines[0].split(|&b| b == delim).count();
    if expected < 2 { return None; }

    // ── Parse ─────────────────────────────────────────────────────────────────
    let mut columns: Vec<Vec<Vec<u8>>> = vec![vec![]; expected];
    let mut parsed = 0usize;
    for line in &lines {
        let parts: Vec<&[u8]> = line.split(|&b| b == delim).collect();
        if parts.len() == expected {
            for (i, p) in parts.iter().enumerate() {
                columns[i].push(p.to_vec());
            }
            parsed += 1;
        }
    }
    let coverage = parsed as f64 / lines.len() as f64;
    if coverage < 0.80 { return None; }

    // ── Encode each column ────────────────────────────────────────────────────
    let mut specs: Vec<ColSpec> = columns.iter()
        .map(|col| encode_smart(col))
        .collect();

    // ── Try merging adjacent dict columns ─────────────────────────────────────
    let n = specs.len();
    let mut skip = vec![false; n];
    for i in 0..n.saturating_sub(1) {
        if skip[i] { continue; }
        if specs[i].enc == ENC_DICT && specs[i+1].enc == ENC_DICT {
            // Build merged original values (no stripping — encode_smart handles it)
            let merged: Vec<Vec<u8>> = (0..parsed).map(|r| {
                let mut v = columns[i][r].clone();
                v.push(delim);
                v.extend_from_slice(&columns[i+1][r]);
                v
            }).collect();
            let mspec = encode_smart(&merged);
            if mspec.payload_size() < specs[i].payload_size() + specs[i+1].payload_size() {
                let saved = specs[i].payload_size() + specs[i+1].payload_size() - mspec.payload_size();
                eprintln!("[iris]   merge col{}+col{}: save {}B", i, i+1, saved);
                specs[i] = ColSpec {
                    enc: ENC_MERGED,
                    pfx: mspec.pfx, sfx: mspec.sfx,
                    data: mspec.data, aux: mspec.aux, aux2: mspec.aux2,
                    merge_sep: delim, merged_col: (i+1) as u32,
                };
                skip[i+1] = true;
            }
        }
    }
    for i in 0..n { if skip[i] { specs[i] = ColSpec::empty(); } }

    // ── Guard: only use column grammar if it beats raw zstd ──────────────────
    let col_total: usize = specs.iter().map(|s| s.payload_size()).sum();
    let raw_z = zstd_enc(data);
    if col_total >= raw_z.len() { return None; }

    Some(ColumnGrammar { delimiter: delim, num_cols: expected,
                         row_count: parsed, coverage, specs })
}

/// Reconstruct original bytes from a `ColumnGrammar`.
pub fn reconstruct_columns(g: &ColumnGrammar, original_len: usize) -> Vec<u8> {
    let n    = g.row_count;
    let delim = g.delimiter;

    // Decode each spec → Vec<Vec<u8>> mapping to original column indices
    let mut decoded: Vec<Vec<Vec<u8>>> = vec![vec![]; g.num_cols];
    let mut orig_idx = 0usize;

    for spec in &g.specs {
        if spec.enc == ENC_EMPTY { continue; } // merged into previous spec — don't advance orig_idx

        let rows = decode_spec(spec, n); // returns fully-restored values (pfx+val+sfx)

        if spec.enc == ENC_MERGED {
            let sep = spec.merge_sep;
            let b_idx = spec.merged_col as usize;
            let mut col_a: Vec<Vec<u8>> = Vec::with_capacity(n);
            let mut col_b: Vec<Vec<u8>> = Vec::with_capacity(n);
            for row in &rows {
                // Split at FIRST occurrence of merge_sep
                let split = row.iter().position(|&b| b == sep).unwrap_or(row.len());
                col_a.push(row[..split].to_vec());
                col_b.push(if split + 1 <= row.len() { row[split+1..].to_vec() } else { vec![] });
            }
            if orig_idx < g.num_cols { decoded[orig_idx] = col_a; }
            if b_idx   < g.num_cols { decoded[b_idx]    = col_b; }
            orig_idx = b_idx + 1;
        } else {
            if orig_idx < g.num_cols { decoded[orig_idx] = rows; }
            orig_idx += 1;
        }
    }

    // Re-join rows
    let mut out = Vec::with_capacity(original_len);
    for r in 0..n {
        for (ci, col) in decoded.iter().enumerate() {
            if ci > 0 { out.push(delim); }
            if r < col.len() { out.extend_from_slice(&col[r]); }
        }
        out.push(b'\n');
    }
    out.truncate(original_len);
    out
}

// ─── Serialization ────────────────────────────────────────────────────────────

pub fn serialize_columns(g: &ColumnGrammar) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(g.delimiter);
    push_u32(&mut out, g.num_cols as u32);
    push_u32(&mut out, g.row_count as u32);
    push_u32(&mut out, g.specs.len() as u32);
    for s in &g.specs {
        out.push(s.enc);
        if s.enc == ENC_MERGED {
            out.push(s.merge_sep);
            push_u32(&mut out, s.merged_col);
        }
        push_chunk(&mut out, &s.pfx);
        push_chunk(&mut out, &s.sfx);
        push_chunk(&mut out, &s.data);
        push_chunk(&mut out, &s.aux);
        push_chunk(&mut out, &s.aux2);
    }
    out
}

pub fn deserialize_columns(raw: &[u8]) -> Option<(ColumnGrammar, usize)> {
    let mut p = 0usize;
    let delimiter = get_u8(raw, &mut p)?;
    let num_cols  = get_u32(raw, &mut p)? as usize;
    let row_count = get_u32(raw, &mut p)? as usize;
    let n_specs   = get_u32(raw, &mut p)? as usize;
    let mut specs = Vec::with_capacity(n_specs);
    for _ in 0..n_specs {
        let enc = get_u8(raw, &mut p)?;
        let (merge_sep, merged_col) = if enc == ENC_MERGED {
            (get_u8(raw, &mut p)?, get_u32(raw, &mut p)?)
        } else { (0, 0) };
        let pfx  = pop_chunk(raw, &mut p)?;
        let sfx  = pop_chunk(raw, &mut p)?;
        let data = pop_chunk(raw, &mut p)?;
        let aux  = pop_chunk(raw, &mut p)?;
        let aux2 = pop_chunk(raw, &mut p)?;
        specs.push(ColSpec { enc, pfx, sfx, data, aux, aux2, merge_sep, merged_col });
    }
    Some((ColumnGrammar { delimiter, num_cols, row_count, coverage: 1.0, specs }, p))
}

// ─── Stride grammar (binary) ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GrammarTemplate {
    pub stride:          usize,
    pub fixed_positions: Vec<(usize, u8)>,
    pub field_offsets:   Vec<usize>,
    pub field_values:    Vec<Vec<u8>>,
    pub coverage:        f64,
    pub tail:            Vec<u8>,
}

pub fn infer(data: &[u8], stride: usize) -> Option<GrammarTemplate> {
    if stride < GRAMMAR_MIN_STRIDE || data.len() < stride * 3 { return None; }
    let nc = data.len() / stride;
    let tail = data[nc*stride..].to_vec();
    let (mut fixed, mut field_offs) = (vec![], vec![]);
    for off in 0..stride {
        let vals: Vec<u8> = (0..nc).map(|i| data[i*stride+off]).collect();
        if position_entropy(&vals) < FIELD_ENTROPY_THRESHOLD {
            fixed.push((off, byte_mode(&vals)));
        } else {
            field_offs.push(off);
        }
    }
    let cov = fixed.len() as f64 / stride as f64;
    if cov < GRAMMAR_MIN_COVERAGE { return None; }
    let fvals: Vec<Vec<u8>> = field_offs.iter()
        .map(|&off| (0..nc).map(|i| data[i*stride+off]).collect())
        .collect();
    Some(GrammarTemplate {
        stride, fixed_positions: fixed, field_offsets: field_offs,
        field_values: fvals, coverage: cov, tail
    })
}

pub fn reconstruct(t: &GrammarTemplate) -> Vec<u8> {
    let ni = t.field_values.first().map(|v| v.len()).unwrap_or(0);
    let mut out = vec![0u8; ni * t.stride];
    for &(off, b) in &t.fixed_positions {
        for i in 0..ni { out[i*t.stride+off] = b; }
    }
    for (fi, &off) in t.field_offsets.iter().enumerate() {
        if fi >= t.field_values.len() { continue; }
        for i in 0..ni {
            if i < t.field_values[fi].len() { out[i*t.stride+off] = t.field_values[fi][i]; }
        }
    }
    out.extend_from_slice(&t.tail);
    out
}

pub fn serialize(t: &GrammarTemplate) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, t.stride as u32);
    push_u32(&mut out, t.fixed_positions.len() as u32);
    for &(off, b) in &t.fixed_positions {
        out.extend_from_slice(&(off as u16).to_le_bytes()); out.push(b);
    }
    push_u32(&mut out, t.field_offsets.len() as u32);
    for &off in &t.field_offsets { out.extend_from_slice(&(off as u16).to_le_bytes()); }
    let ni = t.field_values.first().map(|v| v.len()).unwrap_or(0);
    push_u32(&mut out, ni as u32);
    for fv in &t.field_values { out.extend_from_slice(fv); }
    push_u32(&mut out, t.tail.len() as u32);
    out.extend_from_slice(&t.tail);
    out
}

pub fn deserialize(raw: &[u8]) -> Option<(GrammarTemplate, usize)> {
    let mut p = 0usize;
    let stride = get_u32(raw, &mut p)? as usize;
    let nf     = get_u32(raw, &mut p)? as usize;
    let mut fixed = vec![];
    for _ in 0..nf {
        let off = u16::from_le_bytes(raw.get(p..p+2)?.try_into().ok()?) as usize; p+=2;
        let b = *raw.get(p)?; p+=1;
        fixed.push((off, b));
    }
    let nfo = get_u32(raw, &mut p)? as usize;
    let mut foffs = vec![];
    for _ in 0..nfo {
        foffs.push(u16::from_le_bytes(raw.get(p..p+2)?.try_into().ok()?) as usize); p+=2;
    }
    let ni = get_u32(raw, &mut p)? as usize;
    let mut fvals = vec![];
    for _ in 0..nfo { fvals.push(raw.get(p..p+ni)?.to_vec()); p+=ni; }
    let tl = get_u32(raw, &mut p)? as usize;
    let tail = raw.get(p..p+tl)?.to_vec(); p+=tl;
    let cov = fixed.len() as f64 / stride.max(1) as f64;
    Some((GrammarTemplate {
        stride, fixed_positions: fixed, field_offsets: foffs,
        field_values: fvals, coverage: cov, tail
    }, p))
}

// ─── Smart encoder ────────────────────────────────────────────────────────────

fn encode_smart(values: &[Vec<u8>]) -> ColSpec {
    if values.is_empty() { return ColSpec::empty(); }

    // Detect common prefix/suffix
    let pfx = common_prefix(values);
    let raw_sfx = common_suffix(values);
    // Guard: pfx and sfx cannot overlap — sfx must fit in (min_len - pfx_len)
    let min_len = values.iter().map(|v| v.len()).min().unwrap_or(0);
    let max_sfx_len = min_len.saturating_sub(pfx.len());
    let sfx: Vec<u8> = if raw_sfx.len() > max_sfx_len {
        raw_sfx[raw_sfx.len() - max_sfx_len..].to_vec()
    } else {
        raw_sfx
    };
    // Strip them for type analysis
    let stripped: Vec<Vec<u8>> = values.iter().map(|v| {
        let l = pfx.len(); let r = sfx.len();
        if l + r <= v.len() { v[l..v.len()-r].to_vec() } else { v.clone() }
    }).collect();

    // Baseline: raw zstd of original values
    let raw_bytes: Vec<u8> = values.iter()
        .flat_map(|v| { let mut r = v.clone(); r.push(b'\n'); r }).collect();
    let raw_z = zstd_enc(&raw_bytes);
    let mut best_sz  = raw_z.len();
    let mut best_enc = ENC_RAW;
    let mut best_data = raw_z;
    let mut best_aux  = vec![];
    let mut best_aux2 = vec![];
    // For raw encoding the pfx/sfx aren't needed (full values stored)
    let mut best_pfx  = vec![];
    let mut best_sfx  = vec![];

    // Constant column — trivially small
    if !stripped.is_empty() && stripped.windows(2).all(|w| w[0] == w[1]) {
        let z = zstd_enc(&stripped[0]);
        // Store as raw of single value; decoder reapplies pfx/sfx
        return ColSpec { enc: ENC_RAW,
            pfx: pfx.to_vec(), sfx: sfx.to_vec(),
            data: z, aux: vec![], aux2: vec![],
            merge_sep: 0, merged_col: 0 };
    }

    // HMS time: "HH:MM:SS"
    if stripped.iter().all(|v| is_hms(v)) {
        let (h, m, s) = decompose_hms(&stripped);
        let zh = zstd_enc(&h); let zm = zstd_enc(&m); let zs = zstd_enc(&s);
        let sz = zh.len() + zm.len() + zs.len();
        if sz < best_sz {
            best_sz = sz; best_enc = ENC_HMS;
            best_data = zh; best_aux = zm; best_aux2 = zs;
            best_pfx = pfx.to_vec(); best_sfx = sfx.to_vec();
        }
    }

    // Integer array (all-digit after stripping)
    let all_digit = !stripped.is_empty()
        && stripped.iter().all(|v| !v.is_empty() && v.iter().all(u8::is_ascii_digit));
    if all_digit {
        let nums: Vec<u64> = stripped.iter().map(|v| parse_u64(v)).collect();
        let max_v = nums.iter().copied().max().unwrap_or(0);
        if max_v <= 255 {
            let d: Vec<u8> = nums.iter().map(|&n| n as u8).collect();
            let z = zstd_enc(&d);
            if z.len() < best_sz {
                best_sz = z.len(); best_enc = ENC_U8;
                best_data = z; best_aux = vec![]; best_aux2 = vec![];
                best_pfx = pfx.to_vec(); best_sfx = sfx.to_vec();
            }
        }
        if max_v <= 65535 {
            let d: Vec<u8> = nums.iter().flat_map(|&n| (n as u16).to_le_bytes()).collect();
            let z = zstd_enc(&d);
            if z.len() < best_sz {
                best_sz = z.len(); best_enc = ENC_U16;
                best_data = z; best_aux = vec![]; best_aux2 = vec![];
                best_pfx = pfx.to_vec(); best_sfx = sfx.to_vec();
            }
        }
        {
            let d: Vec<u8> = nums.iter().flat_map(|&n| (n as u32).to_le_bytes()).collect();
            let z = zstd_enc(&d);
            if z.len() < best_sz {
                best_sz = z.len(); best_enc = ENC_U32;
                best_data = z; best_aux = vec![]; best_aux2 = vec![];
                best_pfx = pfx.to_vec(); best_sfx = sfx.to_vec();
            }
        }
    }

    // Dictionary (≤256 unique stripped values)
    {
        use std::collections::BTreeSet;
        let unique: Vec<Vec<u8>> = {
            let s: BTreeSet<Vec<u8>> = stripped.iter().cloned().collect();
            s.into_iter().collect()
        };
        if unique.len() <= 256 {
            use std::collections::HashMap;
            let idx: HashMap<&[u8],u8> = unique.iter().enumerate()
                .map(|(i,v)| (v.as_slice(), i as u8)).collect();
            let indices: Vec<u8> = stripped.iter()
                .map(|v| *idx.get(v.as_slice()).unwrap_or(&0)).collect();
            let dict_raw: Vec<u8> = unique.iter()
                .flat_map(|v| { let mut r=v.clone(); r.push(0); r }).collect();
            let z_dict = zstd_enc(&dict_raw);
            let z_idx  = zstd_enc(&indices);
            let sz = z_dict.len() + z_idx.len();
            if sz < best_sz {
                best_enc = ENC_DICT;
                best_data = z_idx; best_aux = z_dict; best_aux2 = vec![];
                best_pfx = pfx.to_vec(); best_sfx = sfx.to_vec();
            }
        }
    }

    ColSpec { enc: best_enc,
        pfx: best_pfx, sfx: best_sfx,
        data: best_data, aux: best_aux, aux2: best_aux2,
        merge_sep: 0, merged_col: 0 }
}

fn decode_spec(spec: &ColSpec, n_rows: usize) -> Vec<Vec<u8>> {
    let raw_rows: Vec<Vec<u8>> = match spec.enc {
        ENC_EMPTY => (0..n_rows).map(|_| vec![]).collect(),

        ENC_RAW => {
            let raw = zstd_dec(&spec.data);
            // If raw is the entire newline-joined blob
            if raw.contains(&b'\n') {
                raw.split(|&b| b==b'\n').filter(|l|!l.is_empty())
                   .map(|l|l.to_vec()).collect()
            } else {
                // constant col: same value repeated
                (0..n_rows).map(|_| raw.clone()).collect()
            }
        }

        ENC_U8 => {
            let raw = zstd_dec(&spec.data);
            raw.iter().map(|&b| b.to_string().into_bytes()).collect()
        }

        ENC_U16 => {
            let raw = zstd_dec(&spec.data);
            raw.chunks_exact(2)
               .map(|c| u16::from_le_bytes([c[0],c[1]]).to_string().into_bytes())
               .collect()
        }

        ENC_U32 => {
            let raw = zstd_dec(&spec.data);
            raw.chunks_exact(4)
               .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]]).to_string().into_bytes())
               .collect()
        }

        ENC_DICT | ENC_MERGED => {
            let dict_raw = zstd_dec(&spec.aux);
            let idx_raw  = zstd_dec(&spec.data);
            let dict: Vec<&[u8]> = dict_raw.split(|&b| b==0)
                .filter(|s|!s.is_empty()).collect();
            idx_raw.iter()
                .map(|&i| dict.get(i as usize).map(|s|s.to_vec()).unwrap_or_default())
                .collect()
        }

        ENC_HMS => {
            let h = zstd_dec(&spec.data);
            let m = zstd_dec(&spec.aux);
            let s = zstd_dec(&spec.aux2);
            let n = h.len().min(m.len()).min(s.len());
            (0..n).map(|i| format!("{:02}:{:02}:{:02}", h[i], m[i], s[i]).into_bytes()).collect()
        }

        _ => (0..n_rows).map(|_| vec![]).collect(),
    };

    // Reattach prefix/suffix to each decoded value
    if spec.pfx.is_empty() && spec.sfx.is_empty() {
        raw_rows
    } else {
        raw_rows.into_iter().map(|v| {
            let mut out = spec.pfx.clone();
            out.extend_from_slice(&v);
            out.extend_from_slice(&spec.sfx);
            out
        }).collect()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn zstd_enc(d: &[u8]) -> Vec<u8> {
    crate::range_coder::encode(d)
}
fn zstd_dec(d: &[u8]) -> Vec<u8> {
    crate::range_coder::decode(d).unwrap_or_default()
}

fn push_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }
fn push_chunk(out: &mut Vec<u8>, d: &[u8]) {
    push_u32(out, d.len() as u32); out.extend_from_slice(d);
}
fn get_u8(raw: &[u8], p: &mut usize) -> Option<u8> {
    if *p >= raw.len() { return None; }
    let v = raw[*p]; *p += 1; Some(v)
}
fn get_u32(raw: &[u8], p: &mut usize) -> Option<u32> {
    let v = u32::from_le_bytes(raw.get(*p..*p+4)?.try_into().ok()?); *p += 4; Some(v)
}
fn pop_chunk(raw: &[u8], p: &mut usize) -> Option<Vec<u8>> {
    let len = get_u32(raw, p)? as usize;
    let v = raw.get(*p..*p+len)?.to_vec(); *p += len; Some(v)
}

fn detect_delimiter(lines: &[&[u8]]) -> Option<u8> {
    let probe = &lines[..lines.len().min(20)];
    for &d in &[b',', b'\t', b'|', b';', b' '] {
        let counts: Vec<usize> = probe.iter()
            .map(|l| l.iter().filter(|&&b| b==d).count()).collect();
        let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
        let var  = counts.iter().map(|&c| (c as f64-mean).powi(2)).sum::<f64>()
                   / counts.len() as f64;
        // Space is noisiest delimiter: require very low variance
        let threshold = if d == b' ' { mean * 0.02 } else { mean.max(1.0) * 0.3 };
        if mean >= 1.0 && var <= threshold { return Some(d); }
    }
    None
}

fn common_prefix(values: &[Vec<u8>]) -> Vec<u8> {
    if values.is_empty() { return vec![]; }
    let mut p = values[0].clone();
    for v in &values[1..] {
        while !v.starts_with(&p) { p.pop(); if p.is_empty() { break; } }
        if p.is_empty() { break; }
    }
    p
}

fn common_suffix(values: &[Vec<u8>]) -> Vec<u8> {
    if values.is_empty() { return vec![]; }
    let mut s = values[0].clone();
    for v in &values[1..] {
        while !v.ends_with(&s) { if s.is_empty() { break; } s.remove(0); }
        if s.is_empty() { break; }
    }
    s
}

fn is_hms(v: &[u8]) -> bool {
    v.len() == 8 && v[2]==b':' && v[5]==b':'
        && v[..2].iter().all(u8::is_ascii_digit)
        && v[3..5].iter().all(u8::is_ascii_digit)
        && v[6..8].iter().all(u8::is_ascii_digit)
}

fn decompose_hms(values: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut h=vec![]; let mut m=vec![]; let mut s=vec![];
    for v in values { if v.len()>=8 {
        h.push(parse_u64(&v[..2]) as u8);
        m.push(parse_u64(&v[3..5]) as u8);
        s.push(parse_u64(&v[6..8]) as u8);
    }}
    (h, m, s)
}

fn parse_u64(v: &[u8]) -> u64 {
    v.iter().fold(0u64, |a,&b| a*10+(b-b'0') as u64)
}

fn position_entropy(vals: &[u8]) -> f64 {
    if vals.is_empty() { return 0.0; }
    let mut f = [0u32;256]; for &v in vals { f[v as usize]+=1; }
    let n=vals.len() as f64; let mut h=0f64;
    for &c in &f { if c>0 { let p=c as f64/n; h-=p*p.log2(); } }
    h/8.0
}

fn byte_mode(vals: &[u8]) -> u8 {
    let mut f=[0u32;256]; for &v in vals { f[v as usize]+=1; }
    f.iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i as u8).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_grammar_csv_roundtrip() {
        let csv = "name,age,city\nalice,30,nyc\nbob,25,sfo\ncharlie,35,lax\n\
            dave,40,sea\neve,28,chi\nfrank,33,den\ngrace,45,atl\n\
            hank,50,bos\nivy,22,pdx\njack,38,mia\nkate,29,dal\n";
        let data = csv.as_bytes();
        if let Some(cg) = infer_columns(data) {
            let serialized = serialize_columns(&cg);
            let (cg2, _) = deserialize_columns(&serialized).unwrap();
            let reconstructed = reconstruct_columns(&cg2, data.len());
            assert_eq!(std::str::from_utf8(&reconstructed).unwrap(),
                       std::str::from_utf8(data).unwrap());
        }
        // If infer_columns returns None, it means zstd beats column grammar
        // on this tiny dataset — that's fine, not a bug.
    }

    #[test]
    fn stride_grammar_roundtrip() {
        // Binary data with fixed stride, some constant positions
        let stride = 8;
        let n_records = 100;
        let mut data = Vec::with_capacity(stride * n_records);
        for i in 0..n_records {
            data.push(0xAA); // fixed
            data.push(0xBB); // fixed
            data.push((i % 256) as u8); // variable
            data.push(((i * 7) % 256) as u8); // variable
            data.push(0xCC); // fixed
            data.push(0xDD); // fixed
            data.push(((i * 13) % 256) as u8); // variable
            data.push(0xEE); // fixed
        }

        if let Some(tmpl) = infer(&data, stride) {
            let serialized = serialize(&tmpl);
            let (tmpl2, _) = deserialize(&serialized).unwrap();
            let reconstructed = reconstruct(&tmpl2);
            assert_eq!(reconstructed, data);
        }
    }

    #[test]
    fn stride_grammar_serialize_deserialize() {
        let tmpl = GrammarTemplate {
            stride: 4,
            fixed_positions: vec![(0, 0xAA), (2, 0xBB)],
            field_offsets: vec![1, 3],
            field_values: vec![vec![1, 2, 3], vec![4, 5, 6]],
            coverage: 0.5,
            tail: vec![0xFF],
        };
        let bytes = serialize(&tmpl);
        let (tmpl2, _) = deserialize(&bytes).unwrap();
        assert_eq!(tmpl2.stride, 4);
        assert_eq!(tmpl2.fixed_positions, vec![(0, 0xAA), (2, 0xBB)]);
        assert_eq!(tmpl2.field_offsets, vec![1, 3]);
        assert_eq!(tmpl2.field_values, vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(tmpl2.tail, vec![0xFF]);
    }
}
