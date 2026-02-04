use bloomsday::BlockedBloomFilter;
use criterion::{Criterion, criterion_group, criterion_main};
use fastbloom::BloomFilter;
use rand::Rng;
use rand::distr::Alphanumeric;
use sbbf_rs::{ALIGNMENT, BUCKET_SIZE, FilterFn};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::hash::{BuildHasher, Hash, Hasher};
use std::hint::black_box;
use xxhash_rust::xxh64::Xxh64;

#[derive(Clone, Copy)]
struct Xxh64Builder(u64);

impl BuildHasher for Xxh64Builder {
    type Hasher = Xxh64;
    fn build_hasher(&self) -> Self::Hasher {
        Xxh64::new(self.0)
    }
}

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
    pub fn insert_hash(&mut self, h: u64) {
        unsafe {
            self.filter_fn.insert(self.buf, self.num_buckets, h);
        }
    }

    #[inline(always)]
    pub fn contains_hash(&self, h: u64) -> bool {
        unsafe { self.filter_fn.contains(self.buf, self.num_buckets, h) }
    }

    pub fn insert_key<T: Hash>(&mut self, key: &T) {
        let mut hasher = Xxh64::new(0);
        key.hash(&mut hasher);
        self.insert_hash(hasher.finish());
    }

    pub fn contains_key<T: Hash>(&self, key: &T) -> bool {
        let mut hasher = Xxh64::new(0);
        key.hash(&mut hasher);
        self.contains_hash(hasher.finish())
    }
}

impl Drop for SbbfWrapper {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.buf, self.layout);
        }
    }
}

fn bench_bloom_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bloom Filter Comparison (Key-based)");

    let entry_count = 10_000;
    let fpr = 0.01;

    let keys: Vec<String> = (0..entry_count)
        .map(|_| {
            rand::rng()
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect()
        })
        .collect();

    let mut sbbf_filter = SbbfWrapper::new(entry_count, fpr);
    for k in &keys {
        sbbf_filter.insert_key(k);
    }

    let mut blk_filter = BlockedBloomFilter::new(entry_count, fpr);
    for k in &keys {
        blk_filter.insert_key(k);
    }

    let mut fb_filter = BloomFilter::with_false_pos(fpr)
        .hasher(Xxh64Builder(0))
        .expected_items(entry_count);
    for k in &keys {
        fb_filter.insert(k);
    }

    let neg_keys: Vec<String> = (0..1000)
        .map(|_| {
            rand::rng()
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect()
        })
        .collect();
    let pos_keys = &keys[0..1000];

    group.bench_function("SBBF-RS (Negative)", |b| {
        b.iter(|| {
            for k in &neg_keys {
                black_box(sbbf_filter.contains_key(k));
            }
        })
    });

    group.bench_function("Blocked BF (Negative)", |b| {
        b.iter(|| {
            for k in &neg_keys {
                black_box(blk_filter.may_match_key(k));
            }
        })
    });

    group.bench_function("FastBloom (Negative)", |b| {
        b.iter(|| {
            for k in &neg_keys {
                black_box(fb_filter.contains(k));
            }
        })
    });

    group.bench_function("SBBF-RS (Positive)", |b| {
        b.iter(|| {
            for k in pos_keys {
                black_box(sbbf_filter.contains_key(k));
            }
        })
    });

    group.bench_function("Blocked BF (Positive)", |b| {
        b.iter(|| {
            for k in pos_keys {
                black_box(blk_filter.may_match_key(k));
            }
        })
    });

    group.bench_function("FastBloom (Positive)", |b| {
        b.iter(|| {
            for k in pos_keys {
                black_box(fb_filter.contains(k));
            }
        })
    });

    group.finish();
}

fn bench_hash_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bloom Filter Comparison (Hash-based)");

    let entry_count = 10_000;
    let fpr = 0.01;

    let mut rng = rand::rng();
    let hashes: Vec<u64> = (0..entry_count).map(|_| rng.random()).collect();

    let mut sbbf_filter = SbbfWrapper::new(entry_count, fpr);
    for &h in &hashes {
        sbbf_filter.insert_hash(h);
    }

    let mut blk_filter = BlockedBloomFilter::new(entry_count, fpr);
    for &h in &hashes {
        blk_filter.insert_hash(h);
    }

    let neg_hashes: Vec<u64> = (0..1000).map(|_| rng.random()).collect();
    let pos_hashes = &hashes[0..1000];

    group.bench_function("SBBF-RS (Negative)", |b| {
        b.iter(|| {
            for &h in &neg_hashes {
                black_box(sbbf_filter.contains_hash(h));
            }
        })
    });

    group.bench_function("Blocked BF (Negative)", |b| {
        b.iter(|| {
            for &h in &neg_hashes {
                black_box(blk_filter.may_match_hash(h));
            }
        })
    });

    group.bench_function("SBBF-RS (Positive)", |b| {
        b.iter(|| {
            for &h in pos_hashes {
                black_box(sbbf_filter.contains_hash(h));
            }
        })
    });

    group.bench_function("Blocked BF (Positive)", |b| {
        b.iter(|| {
            for &h in pos_hashes {
                black_box(blk_filter.may_match_hash(h));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_bloom_filters, bench_hash_performance);
criterion_main!(benches);
