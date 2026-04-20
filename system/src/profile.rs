//! CompressionProfile — structural/statistical profile of the input.
//!
//! There are now two profiling entry points:
//!
//!   * [`profile_fast`] — sampling-based, bounded to `MAX_SAMPLE_BYTES`
//!     (≤ 64 KB by default). Computes all statistics used by the gate
//!     (entropy, autocorrelation, prediction accuracy, skewness,
//!     local-entropy variance, autocorrelation-peak stability). Does NOT
//!     materialise per-context buckets. Use this for stage-gate decisions
//!     on large files — it runs in O(sample) time and O(1) extra memory.
//!
//!   * [`profile`] — full pass (legacy). Builds context buckets that
//!     context-packing needs. Only call this on the residual that is
//!     actually going to be context-packed, because the buckets cost
//!     `~4·N` bytes.
//!
//! [`build_context_table`] is also exposed for callers who want the
//! buckets without redoing the rest of the profiling work.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct CompressionProfile {
    pub global_entropy:       f64,
    pub global_disorder:      f64,
    pub prediction_accuracy:  f64,
    pub resonance_lags:       Vec<ResonanceLag>,
    pub context_table:        ContextTable,
    pub block_fingerprints:   Vec<u64>,
    pub block_size:           usize,
    pub data_len:             usize,

    // ── Extended statistics (populated by `profile_fast` and `profile`) ──
    /// Mean byte value ∈ [0,255].
    pub byte_mean:            f64,
    /// Population variance of byte values.
    pub byte_variance:        f64,
    /// Pearson moment coefficient of skewness (unitless).
    pub byte_skewness:        f64,
    /// Std-dev of Shannon entropy across sliding 4 KB windows of the sample.
    /// Large values ⇒ heterogeneous data (mixed regions). Small values ⇒
    /// statistically homogeneous data.
    pub local_entropy_stddev: f64,
    /// Stability of the dominant autocorrelation lag across halves of the
    /// sample. 1.0 = identical strongest lag in both halves; 0.0 = different.
    /// Used by the adaptive gate to decide whether resonance is worth running.
    pub autocorr_peak_stability: f64,
    /// Number of bytes inspected to build this profile. For `profile_fast`
    /// this is ≤ [`MAX_SAMPLE_BYTES`]; for `profile` it equals `data_len`.
    pub sample_bytes:         usize,
}

impl CompressionProfile {
    pub fn empty() -> Self {
        Self {
            global_entropy: 0.0, global_disorder: 0.0, prediction_accuracy: 1.0,
            resonance_lags: vec![],
            context_table: ContextTable::new(),
            block_fingerprints: vec![], block_size: BLOCK_SIZE, data_len: 0,
            byte_mean: 0.0, byte_variance: 0.0, byte_skewness: 0.0,
            local_entropy_stddev: 0.0, autocorr_peak_stability: 0.0,
            sample_bytes: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResonanceLag {
    pub lag:      usize,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub struct ContextTable {
    pub buckets:         [Vec<u32>; 256],
    pub active_contexts: Vec<u8>,
}

impl ContextTable {
    pub fn new() -> Self {
        Self { buckets: std::array::from_fn(|_| Vec::new()), active_contexts: Vec::new() }
    }

    /// Approximate memory footprint in bytes. Useful for instrumentation.
    pub fn memory_bytes(&self) -> usize {
        self.buckets.iter().map(|b| b.capacity() * 4).sum::<usize>()
            + self.active_contexts.capacity()
    }
}

// ── StageGate kept for backward compatibility; new code should use crate::gate ──
#[derive(Debug, Clone)]
pub struct StageGate {
    pub resonance:        bool,
    pub grammar:          bool,
    pub prediction_graph: bool,
    pub context_pack:     bool,
    pub passthrough:      bool,
}

impl StageGate {
    pub fn from_profile(p: &CompressionProfile) -> Self {
        crate::gate::AdaptiveGate::default().decide(p).into()
    }
}

// ── Tunables ────────────────────────────────────────────────────────────────
pub const MAX_LAG:    usize = 64;
pub const BLOCK_SIZE: usize = 4096;
const LAG_SAMPLE: usize = 4;    // sample every 4th byte inside a lag window

/// Hard cap on bytes read during a sampled profile. Matches the
/// “Entropy-Based Detection of Structure in Genomic Data” convention of
/// sampling ≤ 65 536 bytes per file for fast heuristic profiling.
pub const MAX_SAMPLE_BYTES: usize = 65_536;

// ─── Fast sampling-based profile ─────────────────────────────────────────────

/// Profile the input using at most [`MAX_SAMPLE_BYTES`] bytes.
///
/// For small files (`data.len() <= MAX_SAMPLE_BYTES`) this is equivalent to
/// reading the whole file. For larger files we read three contiguous windows
/// (head / middle / tail) totalling `MAX_SAMPLE_BYTES`, which gives us
/// representative statistics for heterogeneous data (e.g. a VCF with a big
/// header followed by variant records).
///
/// Does not build context buckets. Call [`build_context_table`] on the
/// data-you-actually-want-to-pack if you need them.
pub fn profile_fast(data: &[u8]) -> CompressionProfile {
    let n = data.len();
    if n == 0 { return CompressionProfile::empty(); }

    let sample = sample_bytes(data);
    profile_core(data, &sample, /*build_buckets=*/ false)
}

/// Legacy full-scan profile. Builds context buckets (≈ 4 bytes per input
/// byte of memory). Prefer [`profile_fast`] for gate decisions and only
/// call this (or [`build_context_table`]) on the residual you’re about to
/// context-pack.
pub fn profile(data: &[u8]) -> CompressionProfile {
    if data.is_empty() { return CompressionProfile::empty(); }
    profile_core(data, &FastSample::full(data), /*build_buckets=*/ true)
}

/// Build only the context buckets (order-1 positional index) for `data`.
/// Extracted so pipeline can defer this work until it knows the ctx-pack
/// route will actually fire.
pub fn build_context_table(data: &[u8]) -> ContextTable {
    let mut ctx = ContextTable::new();
    if data.len() < 2 { return ctx; }

    // Pre-size each bucket to its true count in a first O(n) pass so we
    // do not pay quadratic reallocation cost during push. This also caps
    // total allocations at exactly one Vec growth per context.
    let mut counts = [0u32; 256];
    for w in data.windows(2) { counts[w[0] as usize] += 1; }
    for c in 0..256 {
        let cap = counts[c] as usize;
        if cap > 0 { ctx.buckets[c] = Vec::with_capacity(cap); }
    }
    for i in 1..data.len() {
        let prev = data[i - 1] as usize;
        ctx.buckets[prev].push(i as u32);
    }
    ctx.active_contexts = (0u8..=255)
        .filter(|&c| !ctx.buckets[c as usize].is_empty())
        .collect();
    ctx
}

// ── Sampling support ─────────────────────────────────────────────────────────

/// A view into `data` as three contiguous windows. Storing ranges (not a
/// copy) lets `profile_core` walk the sample with full context for
/// autocorrelation without allocating.
struct FastSample {
    ranges: [(usize, usize); 3], // (start, end) — end exclusive
    total:  usize,
}

impl FastSample {
    fn full(data: &[u8]) -> Self {
        Self { ranges: [(0, data.len()), (0, 0), (0, 0)], total: data.len() }
    }
    fn len(&self) -> usize { self.total }
    fn iter_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.ranges.iter().copied().filter(|(s, e)| e > s)
    }
}

fn sample_bytes(data: &[u8]) -> FastSample {
    let n = data.len();
    if n <= MAX_SAMPLE_BYTES {
        return FastSample::full(data);
    }
    // Three equal windows: head, middle, tail.
    let w = MAX_SAMPLE_BYTES / 3;
    let head = (0, w);
    let mid_start = n / 2 - w / 2;
    let mid = (mid_start, mid_start + w);
    // Last window catches the rounding slack so total ≤ MAX_SAMPLE_BYTES.
    let last_w = MAX_SAMPLE_BYTES - 2 * w;
    let tail = (n - last_w, n);
    FastSample {
        ranges: [head, mid, tail],
        total:  MAX_SAMPLE_BYTES,
    }
}

// ─── Core profile routine (sample-aware) ─────────────────────────────────────

fn profile_core(data: &[u8], sample: &FastSample, build_buckets: bool) -> CompressionProfile {
    let n = data.len();

    // Byte histogram over the sample.
    let mut freq = [0u64; 256];
    let mut sample_count: u64 = 0;
    for (s, e) in sample.iter_ranges() {
        for &b in &data[s..e] { freq[b as usize] += 1; sample_count += 1; }
    }
    let global_entropy = shannon_entropy_normalised(&freq, sample_count);
    let (byte_mean, byte_variance, byte_skewness) = byte_moments(&freq, sample_count);

    // Autocorrelation MSD (wrapping-i8) per lag over the sample. We walk
    // each contiguous range separately and only count intra-range pairs.
    let mut acorr = [0i64; MAX_LAG + 1];
    let mut acnt  = [0u64; MAX_LAG + 1];
    let mut acorr_h1 = [0i64; MAX_LAG + 1]; // first half of sample
    let mut acnt_h1  = [0u64; MAX_LAG + 1];
    let mut acorr_h2 = [0i64; MAX_LAG + 1]; // second half
    let mut acnt_h2  = [0u64; MAX_LAG + 1];

    let mut sample_seen: usize = 0;
    let half = sample.len() / 2;

    for (s, e) in sample.iter_ranges() {
        let win = &data[s..e];
        for i in 1..win.len() {
            if i % LAG_SAMPLE == 0 {
                let is_first_half = sample_seen + i < half;
                let lag_max = MAX_LAG.min(i);
                for lag in 1..=lag_max {
                    let diff = win[i].wrapping_sub(win[i - lag]) as i8 as i64;
                    let d2   = diff * diff;
                    acorr[lag] += d2; acnt[lag] += 1;
                    if is_first_half {
                        acorr_h1[lag] += d2; acnt_h1[lag] += 1;
                    } else {
                        acorr_h2[lag] += d2; acnt_h2[lag] += 1;
                    }
                }
            }
        }
        sample_seen += win.len();
    }

    // Prediction accuracy (running |Δbyte|) over the sample.
    let mut running_surprise = 0.5f64;
    let mut prev_byte = 0.5f64;
    let mut first = true;
    for (s, e) in sample.iter_ranges() {
        for &b in &data[s..e] {
            let actual = b as f64 / 255.0;
            if !first {
                let surprise = (actual - prev_byte).abs();
                running_surprise = running_surprise * 0.995 + surprise * 0.005;
            }
            prev_byte = actual;
            first = false;
        }
    }
    let prediction_accuracy = 1.0 - running_surprise.min(1.0);

    // Local entropy variance across 4 KB sub-windows of the sample.
    let local_entropy_stddev = local_entropy_stddev(data, sample);

    // Resonance lags (overall + peak stability).
    let rlags_all = lags_from_msd(&acorr, &acnt);
    let rlags_h1  = lags_from_msd(&acorr_h1, &acnt_h1);
    let rlags_h2  = lags_from_msd(&acorr_h2, &acnt_h2);
    let autocorr_peak_stability =
        peak_stability(rlags_all.first(), rlags_h1.first(), rlags_h2.first());

    let global_disorder = estimate_disorder(data);

    // Context table (only if requested; otherwise empty).
    let context_table = if build_buckets { build_context_table(data) }
                        else             { ContextTable::new() };

    CompressionProfile {
        global_entropy,
        global_disorder,
        prediction_accuracy,
        resonance_lags: rlags_all,
        context_table,
        block_fingerprints: vec![], // no longer populated here; pipeline
                                    // computes SimHashes on demand
        block_size: BLOCK_SIZE,
        data_len: n,
        byte_mean, byte_variance, byte_skewness,
        local_entropy_stddev,
        autocorr_peak_stability,
        sample_bytes: sample.len(),
    }
}

// ─── Statistics helpers ──────────────────────────────────────────────────────

pub fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut freq = [0u64; 256];
    let step = (data.len() / MAX_SAMPLE_BYTES).max(1);
    let mut count = 0u64;
    for &b in data.iter().step_by(step) { freq[b as usize] += 1; count += 1; }
    shannon_entropy_normalised(&freq, count)
}

fn shannon_entropy_normalised(freq: &[u64; 256], count: u64) -> f64 {
    if count == 0 { return 0.0; }
    let n = count as f64;
    let mut h = 0.0f64;
    for &f in freq {
        if f > 0 { let p = f as f64 / n; h -= p * p.log2(); }
    }
    h / 8.0
}

fn byte_moments(freq: &[u64; 256], count: u64) -> (f64, f64, f64) {
    if count == 0 { return (0.0, 0.0, 0.0); }
    let n = count as f64;
    let mut mean = 0.0f64;
    for (v, &f) in freq.iter().enumerate() {
        mean += (v as f64) * (f as f64);
    }
    mean /= n;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for (v, &f) in freq.iter().enumerate() {
        let d = v as f64 - mean;
        m2 += f as f64 * d * d;
        m3 += f as f64 * d * d * d;
    }
    m2 /= n; m3 /= n;
    let variance = m2;
    let skewness = if m2 > 1e-12 { m3 / m2.powf(1.5) } else { 0.0 };
    (mean, variance, skewness)
}

/// Std-dev of byte-entropy computed on sliding 4 KB windows within the sample.
fn local_entropy_stddev(data: &[u8], sample: &FastSample) -> f64 {
    const WIN: usize = 4096;
    let mut entropies: Vec<f64> = Vec::new();
    for (s, e) in sample.iter_ranges() {
        let mut p = s;
        while p + WIN <= e {
            entropies.push(byte_entropy(&data[p..p + WIN]));
            p += WIN;
        }
    }
    if entropies.len() < 2 { return 0.0; }
    let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let var  = entropies.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / entropies.len() as f64;
    var.sqrt()
}

fn lags_from_msd(acorr: &[i64], acnt: &[u64]) -> Vec<ResonanceLag> {
    // Baseline MSD for wrapping i8: uniform bytes → E[(a-b as i8)²]
    // For uniform [0,255], wrapping difference is uniform [-128,127].
    // E[x²] for uniform [-128,127] ≈ 128²/3 ≈ 5461.
    let baseline_msd = 5461i64;
    let mut rlags: Vec<ResonanceLag> = (1..acorr.len())
        .filter(|&l| acnt[l] > 0)
        .map(|lag| {
            let msd      = acorr[lag] / acnt[lag] as i64;
            let strength = 1.0 - (msd as f64 / baseline_msd as f64).clamp(0.0, 1.0);
            ResonanceLag { lag, strength }
        })
        .filter(|r| r.strength > 0.10)
        .collect();
    rlags.sort_by(|a, b|
        b.strength.partial_cmp(&a.strength).unwrap_or(Ordering::Equal)
            .then(a.lag.cmp(&b.lag))
    );
    rlags.truncate(8);
    rlags
}

fn peak_stability(
    all: Option<&ResonanceLag>,
    h1: Option<&ResonanceLag>,
    h2: Option<&ResonanceLag>,
) -> f64 {
    match (all, h1, h2) {
        (Some(a), Some(p), Some(q)) => {
            let lag_match = (p.lag == a.lag) as u8 + (q.lag == a.lag) as u8;
            let base = lag_match as f64 / 2.0;
            // Pull toward 0 if strengths diverge wildly.
            let s_diff = (p.strength - q.strength).abs();
            (base * (1.0 - s_diff).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        }
        _ => 0.0,
    }
}

fn estimate_disorder(data: &[u8]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mut state: u64 = 0xdeadbeef ^ data.len() as u64;
    let n = data.len();
    let mut inv = 0usize;
    let pairs = 64.min(n * (n - 1) / 2);
    for _ in 0..pairs {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = (state >> 33) as usize % n;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = (state >> 33) as usize % n;
        let (lo, hi) = (a.min(b), a.max(b));
        if lo != hi && data[lo] > data[hi] { inv += 1; }
    }
    inv as f64 / pairs as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_profile_small_matches_full_profile() {
        let data: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();
        let fast = profile_fast(&data);
        let full = profile(&data);
        assert!((fast.global_entropy - full.global_entropy).abs() < 1e-6);
        assert_eq!(fast.data_len, full.data_len);
    }

    #[test]
    fn context_table_matches_reference() {
        // "abcabcabc" — context = previous byte. After every 'a' comes a 'b'
        // at indices 1, 4, 7; those indices are what bucket['a'] records.
        let data: Vec<u8> = b"abcabcabc".to_vec();
        let ct = build_context_table(&data);
        assert_eq!(ct.buckets[b'a' as usize], vec![1, 4, 7]);
        assert_eq!(ct.buckets[b'b' as usize], vec![2, 5, 8]);
    }

    #[test]
    fn sampled_profile_is_bounded() {
        let big: Vec<u8> = (0..1_000_000u32).map(|i| (i * 13) as u8).collect();
        let p = profile_fast(&big);
        assert!(p.sample_bytes <= MAX_SAMPLE_BYTES);
        assert_eq!(p.data_len, big.len());
    }

    #[test]
    fn peak_stability_detects_consistent_lag() {
        let data: Vec<u8> = (0..16384u32).map(|i| match i % 4 {
            0 => 0x10, 1 => 0x20, 2 => 0x30, _ => 0x40,
        }).collect();
        let p = profile_fast(&data);
        assert!(!p.resonance_lags.is_empty());
        assert_eq!(p.resonance_lags[0].lag, 4);
        assert!(p.autocorr_peak_stability > 0.5);
    }
}
