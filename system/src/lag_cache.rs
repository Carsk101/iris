//! Lag caching for resonance.
//!
//! Lag detection (autocorrelation scan) is the expensive part of the
//! profiler on large files. For data with stable periodicity (fixed-stride
//! records, RNA-seq coverage arrays, packed image planes, etc.) the
//! dominant lag is the same across all chunks of the file, and re-running
//! a full autocorrelation scan after every resonance pass is wasted work.
//!
//! `LagCache` records lags we have already seen (with their observed
//! strength) and offers two cheap primitives:
//!
//!   1. [`LagCache::probe`] — given a buffer, test whether a cached
//!      candidate lag still yields a good residual MSD on a *sample* of
//!      the buffer. O(sample_bytes), not O(buffer).
//!   2. [`LagCache::note`] — record a lag that was just used successfully.
//!
//! Callers use it as: *first probe the cache; on miss, run the full
//! profiler.* This keeps decompression deterministic because we still
//! store the actual lag value in the container header — the cache is a
//! performance hint, not part of the on-disk format.

use crate::profile::{ResonanceLag, MAX_LAG};

/// How many bytes to scan when probing a candidate lag against a buffer.
/// 16 KB gives a stable MSD estimate for any realistic period ≤ 64.
pub const PROBE_WINDOW: usize = 16_384;

/// Strength threshold below which we refuse to reuse a cached lag and
/// force a full re-profile. Matches `RESONANCE_MIN_STRENGTH` in
/// `resonance.rs` but kept here so the cache module is self-contained.
pub const REUSE_MIN_STRENGTH: f64 = 0.20;

/// Baseline MSD for wrapping-i8 differences of uniform bytes — same
/// constant used in `profile.rs`.
const BASELINE_MSD: i64 = 5461;

#[derive(Debug, Clone)]
pub struct LagCache {
    /// Ordered by insertion (most recent last). Duplicates are allowed
    /// because stored strengths are observation-specific.
    entries: Vec<ResonanceLag>,
    /// Rolling count of cache hits / probes. Useful for instrumentation.
    pub hits:   u64,
    pub misses: u64,
}

impl LagCache {
    pub fn new() -> Self {
        Self { entries: Vec::new(), hits: 0, misses: 0 }
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Record an observed `(lag, strength)` pair. Lags that are already
    /// present with a lower strength are upgraded; otherwise appended.
    pub fn note(&mut self, lag: usize, strength: f64) {
        if let Some(entry) = self.entries.iter_mut().find(|l| l.lag == lag) {
            if strength > entry.strength { entry.strength = strength; }
        } else {
            self.entries.push(ResonanceLag { lag, strength });
        }
        // Keep cache tiny — only the strongest handful of lags matter.
        self.entries.sort_by(|a, b|
            b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal)
        );
        self.entries.truncate(8);
    }

    /// Test every cached lag against `data` using a sampling probe and
    /// return the best candidate whose strength on this buffer meets
    /// `REUSE_MIN_STRENGTH`. Returns `None` on cache miss.
    pub fn probe(&mut self, data: &[u8]) -> Option<ResonanceLag> {
        if self.entries.is_empty() || data.len() < 2 * MAX_LAG {
            self.misses += 1;
            return None;
        }

        let mut best: Option<ResonanceLag> = None;
        for e in &self.entries {
            if e.lag == 0 || e.lag >= data.len() { continue; }
            let s = probe_lag(data, e.lag);
            if s >= REUSE_MIN_STRENGTH {
                if best.as_ref().map_or(true, |b| s > b.strength) {
                    best = Some(ResonanceLag { lag: e.lag, strength: s });
                }
            }
        }

        match &best {
            Some(_) => self.hits   += 1,
            None    => self.misses += 1,
        }
        best
    }
}

impl Default for LagCache {
    fn default() -> Self { Self::new() }
}

/// Quick autocorrelation probe for a single lag over a bounded window of
/// `data`. Cost: O(PROBE_WINDOW).
pub fn probe_lag(data: &[u8], lag: usize) -> f64 {
    if lag == 0 || lag >= data.len() { return 0.0; }
    let probe_len = PROBE_WINDOW.min(data.len());
    // Pick the middle of the buffer so we don’t get misled by a
    // header/prefix that may have different statistics.
    let start = (data.len() - probe_len) / 2;
    let end   = start + probe_len;

    let mut sum: i64 = 0;
    let mut cnt: i64 = 0;
    let mut i = start + lag;
    while i < end {
        let d = data[i].wrapping_sub(data[i - lag]) as i8 as i64;
        sum += d * d;
        cnt += 1;
        i += 4; // same LAG_SAMPLE stride as profile.rs
    }
    if cnt == 0 { return 0.0; }
    let msd = sum / cnt;
    (1.0 - (msd as f64 / BASELINE_MSD as f64)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_periodic_data() {
        let data: Vec<u8> = (0..16_384u32).map(|i| match i % 4 {
            0 => 0x10, 1 => 0x20, 2 => 0x30, _ => 0x40,
        }).collect();
        let s = probe_lag(&data, 4);
        assert!(s > 0.9, "expected strong lag-4 peak, got {}", s);
    }

    #[test]
    fn probe_random_data() {
        let mut s: u64 = 0xbeef;
        let data: Vec<u8> = (0..16_384).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u8
        }).collect();
        let st = probe_lag(&data, 4);
        assert!(st < 0.25, "random data should not trip the probe, got {}", st);
    }

    #[test]
    fn cache_hit_after_note() {
        let data: Vec<u8> = (0..16_384u32).map(|i| (i % 8) as u8 * 0x20).collect();
        let mut cache = LagCache::new();
        cache.note(8, 0.9);
        let hit = cache.probe(&data);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().lag, 8);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn cache_miss_on_unrelated_data() {
        let mut s: u64 = 0xabc;
        let data: Vec<u8> = (0..16_384).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u8
        }).collect();
        let mut cache = LagCache::new();
        cache.note(4, 0.8);
        cache.note(8, 0.7);
        let hit = cache.probe(&data);
        assert!(hit.is_none());
        assert_eq!(cache.misses, 1);
    }
}
