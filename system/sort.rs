/// VisionSort — adapted for iris compression pipeline
/// Treats sorting as a perception problem: builds a probabilistic model
/// during the sort itself, routing elements based on entropy classification.
/// Returns sorted bytes + permutation index for lossless reconstruction.

const MINRUN: usize = 64;
const TRIVIAL: usize = MINRUN * 2;
const SAMPLE_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Route {
    NearlyFree,   // nearly sorted, min-cost verify pass
    Verify,       // low entropy, insertion sort
    PlacementSort,// medium entropy, guided placement
    FullSort,     // high entropy, full merge
}

pub struct SortResult {
    pub sorted: Vec<u8>,
    pub permutation: Vec<u32>,
    pub route: Route,
    pub entropy: f64,
}

impl SortResult {
    /// Returns which permutation regions are high-confidence (nearly deterministic).
    /// Used by the container to skip storing trivially-recoverable index segments.
    pub fn cheap_regions(&self) -> Vec<(usize, usize)> {
        match self.route {
            Route::NearlyFree => {
                // Nearly the entire permutation is recoverable — only store deltas
                vec![]
            }
            Route::Verify => {
                // Short runs are deterministic; only store inter-run boundaries
                find_run_boundaries(&self.sorted)
            }
            _ => vec![],
        }
    }
}

pub fn vision_sort(data: &[u8]) -> SortResult {
    if data.is_empty() {
        return SortResult {
            sorted: vec![],
            permutation: vec![],
            route: Route::NearlyFree,
            entropy: 0.0,
        };
    }

    // Build original index
    let mut indexed: Vec<(u8, u32)> = data
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i as u32))
        .collect();

    let entropy = estimate_entropy(data);
    let route = classify(data, entropy);

    match route {
        Route::NearlyFree | Route::Verify => {
            insertion_sort_indexed(&mut indexed);
        }
        Route::PlacementSort => {
            placement_sort_indexed(&mut indexed, entropy);
        }
        Route::FullSort => {
            merge_sort_indexed(&mut indexed);
        }
    }

    let sorted: Vec<u8> = indexed.iter().map(|(b, _)| *b).collect();
    let permutation: Vec<u32> = indexed.iter().map(|(_, i)| *i).collect();

    SortResult { sorted, permutation, route, entropy }
}

/// Reconstruct original byte order from sorted bytes + permutation index
pub fn unsort(sorted: &[u8], permutation: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; sorted.len()];
    for (sorted_idx, &orig_idx) in permutation.iter().enumerate() {
        out[orig_idx as usize] = sorted[sorted_idx];
    }
    out
}

fn estimate_entropy(data: &[u8]) -> f64 {
    let step = (data.len() / SAMPLE_SIZE).max(1);
    let mut freq = [0u32; 256];
    let mut count = 0usize;

    for &b in data.iter().step_by(step).take(SAMPLE_SIZE) {
        freq[b as usize] += 1;
        count += 1;
    }

    if count == 0 { return 0.0; }
    let n = count as f64;
    let mut h = 0.0f64;
    for &f in freq.iter() {
        if f > 0 {
            let p = f as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

fn classify(data: &[u8], entropy: f64) -> Route {
    if data.len() <= TRIVIAL {
        return Route::Verify;
    }

    // Sample disorder: count inversions in a window
    let window = &data[..data.len().min(256)];
    let inversions = window.windows(2).filter(|w| w[0] > w[1]).count();
    let disorder = inversions as f64 / (window.len() - 1) as f64;

    match (disorder < 0.1, entropy < 4.0) {
        (true, true)  => Route::NearlyFree,
        (true, false) => Route::Verify,
        (false, true) => Route::PlacementSort,
        (false, false) => Route::FullSort,
    }
}

fn insertion_sort_indexed(data: &mut [(u8, u32)]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1].0 > key.0 {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

fn placement_sort_indexed(data: &mut [(u8, u32)], entropy: f64) {
    // Bucket by estimated distribution, then insertion sort within buckets
    let n_buckets = ((entropy * 8.0) as usize).clamp(4, 64);
    let mut buckets: Vec<Vec<(u8, u32)>> = vec![Vec::new(); n_buckets];

    for &item in data.iter() {
        let bucket = (item.0 as usize * n_buckets) / 256;
        buckets[bucket].push(item);
    }

    let mut idx = 0;
    for mut bucket in buckets {
        insertion_sort_indexed(&mut bucket);
        for item in bucket {
            data[idx] = item;
            idx += 1;
        }
    }
}

fn merge_sort_indexed(data: &mut [(u8, u32)]) {
    let n = data.len();
    if n <= MINRUN {
        insertion_sort_indexed(data);
        return;
    }

    // Bottom-up merge with run detection
    let mut run_size = MINRUN;
    while run_size < n {
        let mut left = 0;
        while left < n {
            let mid = (left + run_size).min(n);
            let right = (left + 2 * run_size).min(n);
            if mid < right {
                merge_indexed(data, left, mid, right);
            }
            left += 2 * run_size;
        }
        run_size *= 2;
    }
}

fn merge_indexed(data: &mut [(u8, u32)], left: usize, mid: usize, right: usize) {
    let left_half = data[left..mid].to_vec();
    let right_half = data[mid..right].to_vec();

    let (mut i, mut j, mut k) = (0, 0, left);

    while i < left_half.len() && j < right_half.len() {
        if left_half[i].0 <= right_half[j].0 {
            data[k] = left_half[i]; i += 1;
        } else {
            data[k] = right_half[j]; j += 1;
        }
        k += 1;
    }
    while i < left_half.len() { data[k] = left_half[i]; i += 1; k += 1; }
    while j < right_half.len() { data[k] = right_half[j]; j += 1; k += 1; }
}

fn find_run_boundaries(sorted: &[u8]) -> Vec<(usize, usize)> {
    let mut boundaries = vec![];
    let mut start = 0;
    for i in 1..sorted.len() {
        if sorted[i] != sorted[i - 1] {
            if i - start > MINRUN {
                boundaries.push((start, i));
            }
            start = i;
        }
    }
    boundaries
}
