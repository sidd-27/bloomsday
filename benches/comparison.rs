use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng; 
use rand::distr::Alphanumeric; 
use xxhash_rust::xxh64::xxh64;
use sbbf_rs::{FilterFn, ALIGNMENT, BUCKET_SIZE};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use fast_blocked_bloom::BlockedBloomFilter;

// =========================================================================
// 1. SBBF-RS WRAPPER (With Proper Alignment)
// =========================================================================

pub struct SbbfWrapper {
    filter_fn: FilterFn,
    buf: *mut u8,
    layout: Layout,
    num_buckets: usize,
}

impl SbbfWrapper {
    pub fn new(entries: usize, fpr: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let bits_per_key = ((-1.0 * fpr.ln()) / (ln2 * ln2)).ceil() as usize;
        let num_buckets = (entries * bits_per_key + 255) / 256;
        let buf_size = num_buckets * BUCKET_SIZE;
        
        // Ensure ALIGNMENT (64 bytes)
        let layout = Layout::from_size_align(buf_size, ALIGNMENT).unwrap();
        let buf = unsafe { alloc_zeroed(layout) };
        
        if buf.is_null() {
            panic!("Allocation failed");
        }
        
        Self {
            filter_fn: FilterFn::new(),
            buf,
            layout,
            num_buckets,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, h: u64) {
        unsafe {
            self.filter_fn.insert(self.buf, self.num_buckets, h);
        }
    }

    #[inline(always)]
    pub fn contains(&self, h: u64) -> bool {
        unsafe {
            self.filter_fn.contains(self.buf, self.num_buckets, h)
        }
    }
}

impl Drop for SbbfWrapper {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.buf, self.layout);
        }
    }
}

// =========================================================================
// 2. BENCHMARK EXECUTION
// =========================================================================

fn bench_bloom_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bloom Filter Comparison");
    
    let entry_count = 1_000_000;
    let fpr = 0.01;
    
    // --- 1. Setup Data ---
    // Generate Random Keys
    let keys: Vec<String> = (0..entry_count)
        .map(|_| rand::rng().sample_iter(&Alphanumeric).take(16).map(char::from).collect())
        .collect();
    
    // Create Hashes
    let hashes_blk: Vec<u64> = keys.iter().map(|k| xxh64(k.as_bytes(), 0)).collect();

    // Build Filters
    let mut sbbf_filter = SbbfWrapper::new(entry_count, fpr);
    for &h in &hashes_blk {
        sbbf_filter.insert(h);
    }

    let mut blk_filter = BlockedBloomFilter::new(entry_count, fpr);
    for &h in &hashes_blk {
        blk_filter.insert_hash(h);
    }

    // --- 2. Prepare Test Sets ---
    
    // Set A: Negative (Keys that do NOT exist)
    let neg_keys: Vec<String> = (0..1000) 
        .map(|_| rand::rng().sample_iter(&Alphanumeric).take(16).map(char::from).collect())
        .collect();
    let neg_hashes_blk: Vec<u64> = neg_keys.iter().map(|k| xxh64(k.as_bytes(), 0)).collect();

    // Set B: Positive (Keys that DO exist)
    let pos_hashes_blk = &hashes_blk[0..1000];

    // --- 3. Run Benchmarks ---

    // A. Negative Lookups (Item not present)
    group.bench_function("SBBF-RS (Negative)", |b| {
        b.iter(|| {
            for &h in &neg_hashes_blk {
                black_box(sbbf_filter.contains(h));
            }
        })
    });

    group.bench_function("Blocked BF (Negative)", |b| {
        b.iter(|| {
            for &h in &neg_hashes_blk {
                black_box(blk_filter.may_match_hash(h));
            }
        })
    });

    // B. Positive Lookups (Item IS present)
    group.bench_function("SBBF-RS (Positive)", |b| {
        b.iter(|| {
            for &h in pos_hashes_blk {
                black_box(sbbf_filter.contains(h));
            }
        })
    });

    group.bench_function("Blocked BF (Positive)", |b| {
        b.iter(|| {
            for &h in pos_hashes_blk {
                black_box(blk_filter.may_match_hash(h));
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_bloom_filters);
criterion_main!(benches);
