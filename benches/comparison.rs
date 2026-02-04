use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng; 
use sbbf_rs::{FilterFn, ALIGNMENT, BUCKET_SIZE};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use bloomsday::BlockedBloomFilter;

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

fn bench_hash_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blocked Bloom Filter Hash Lookup");
    
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

    group.bench_function("Bloomsday (Negative)", |b| {
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

    group.bench_function("Bloomsday (Positive)", |b| {
        b.iter(|| {
            for &h in pos_hashes {
                black_box(blk_filter.may_match_hash(h));
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_hash_performance);
criterion_main!(benches);