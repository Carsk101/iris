//! CompressionProfile — single read-only pass over data feeding all four stages.

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
}

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
        let e = p.global_entropy;
        if e > 0.97 {
            return Self { resonance:false, grammar:false, prediction_graph:false,
                          context_pack:false, passthrough:true };
        }
        Self {
            resonance:        e < 0.92,
            grammar:          p.prediction_accuracy > 0.45 && e < 0.80,
            prediction_graph: p.block_fingerprints.len() > 1,
            context_pack:     e < 0.97,
            passthrough:      false,
        }
    }
}

const MAX_LAG:    usize = 64;
const BLOCK_SIZE: usize = 4096;
const LAG_SAMPLE: usize = 4;    // sample every 4th byte

pub fn profile(data: &[u8]) -> CompressionProfile {
    let n = data.len();
    if n == 0 {
        return CompressionProfile {
            global_entropy: 0.0, global_disorder: 0.0,
            prediction_accuracy: 1.0, resonance_lags: vec![],
            context_table: ContextTable::new(),
            block_fingerprints: vec![], block_size: BLOCK_SIZE, data_len: 0,
        };
    }

    let global_entropy  = byte_entropy(data);
    let global_disorder = estimate_disorder(data);

    // Autocorrelation — WRAPPING i8 arithmetic.
    // Signed wrapping difference maps 0→127 and -128→-1, giving fair weight
    // to all cyclic distances. This correctly ranks lag=1 highest for
    // nearly-sequential data (residuals ≈ 1, not -255).
    let mut acorr  = [0i64; MAX_LAG + 1];
    let mut acnt   = [0u64; MAX_LAG + 1];
    let mut ctx    = ContextTable::new();
    let mut fps: Vec<u64> = Vec::new();

    let mut running_surprise = 0.5f64;
    let mut prev_byte = data[0] as f64 / 255.0;

    let mut block_start = 1usize; // profile fingerprints start at byte 1

    for i in 1..n {
        let byte = data[i];
        let prev = data[i - 1];

        ctx.buckets[prev as usize].push(i as u32);

        // Wrapping i8 difference: (byte - prev) as i8 treats the byte ring
        // symmetrically, so lag=1 on delta-1 data gives diff=1 everywhere.
        if i % LAG_SAMPLE == 0 {
            for lag in 1..=MAX_LAG.min(i) {
                let diff = byte.wrapping_sub(data[i - lag]) as i8 as i64;
                acorr[lag] += diff * diff;
                acnt[lag]  += 1;
            }
        }

        let actual   = byte as f64 / 255.0;
        let surprise = (actual - prev_byte).abs();
        running_surprise = running_surprise * 0.995 + surprise * 0.005;
        prev_byte = actual;

        if (i - block_start + 1) >= BLOCK_SIZE || i == n - 1 {
            // Profile fingerprints are only used for gate decisions (block count),
            // not for similarity. Push a simple hash for counting purposes.
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in &data[block_start..=i] {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            fps.push(h);
            block_start = i + 1;
        }
    }

    ctx.active_contexts = (0u8..=255)
        .filter(|&c| !ctx.buckets[c as usize].is_empty())
        .collect();

    // Baseline MSD for wrapping i8: uniform bytes → E[(a-b as i8)²]
    // For uniform [0,255], wrapping difference is uniform [-128,127].
    // E[x²] for uniform [-128,127] ≈ 128²/3 ≈ 5461.
    let baseline_msd = 5461i64;
    let mut rlags: Vec<ResonanceLag> = (1..=MAX_LAG)
        .filter(|&l| acnt[l] > 0)
        .map(|lag| {
            let msd      = acorr[lag] / acnt[lag] as i64;
            let strength = 1.0 - (msd as f64 / baseline_msd as f64).clamp(0.0, 1.0);
            ResonanceLag { lag, strength }
        })
        .filter(|r| r.strength > 0.10)
        .collect();

    // Sort by strength desc, break ties by lag asc (shorter lags preferred)
    rlags.sort_by(|a, b|
        b.strength.partial_cmp(&a.strength).unwrap()
            .then(a.lag.cmp(&b.lag))
    );
    rlags.truncate(8);

    let prediction_accuracy = 1.0 - running_surprise.min(1.0);

    CompressionProfile {
        global_entropy, global_disorder, prediction_accuracy,
        resonance_lags: rlags,
        context_table: ctx,
        block_fingerprints: fps,
        block_size: BLOCK_SIZE,
        data_len: n,
    }
}

pub fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut freq = [0u64; 256];
    let step = (data.len() / 65536).max(1);
    let mut count = 0u64;
    for &b in data.iter().step_by(step) { freq[b as usize] += 1; count += 1; }
    let n = count as f64;
    let mut h = 0.0f64;
    for &f in &freq { if f > 0 { let p = f as f64 / n; h -= p * p.log2(); } }
    h / 8.0
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
