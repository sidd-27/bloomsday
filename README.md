# bloomsday

A high-performance cache-line blocked Bloom filter optimized for modern CPUs.

This is a Rust implementation of the [Parquet Split Block Bloom Filter (SBBF)](https://github.com/apache/parquet-format/blob/master/BloomFilter.md) format, designed for high-throughput membership queries with excellent cache locality.

## Features

- **In compilers we trust**: Instead of brittle, explicit SIMD intrinsics, this library uses compiler-friendly loops that allow LLVM to auto-vectorize the code into efficient SIMD instructions (like `vpand`, `vpor`, `vpsllv`).
- **High Performance**: ~2.3x faster than `sbbf-rs` (Split Block Bloom Filter) in benchmarks.

> **Note**: This library is fast **only** when the compiler is allowed to auto-vectorize the code properly. You should compile with appropriate target features (e.g., `-C target-cpu=native` or `-C target-feature=+avx2`). Without these, the compiler may fallback to scalar instructions, significantly reducing performance.
- **Built-in Hashing**: Includes an easy-to-use API for arbitrary keys using `xxHash` (xxh64).
- **Zero Dependencies**: Core library is lightweight (only `std` and `xxhash-rust`).

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
bloomsday = "0.1.0"
```

### Optional Features

- **serde**: Enables `Serialize` and `Deserialize` support for `BlockedBloomFilter`.

```toml
[dependencies]
bloomsday = { version = "0.1.0", features = ["serde"] }
```

### Example

#### Using the High-Level API (Recommended)
This uses the internal `xxHash` implementation, which is very fast and robust.

```rust
use bloomsday::BlockedBloomFilter;

fn main() {
    let entries = 1_000_000;
    let fpr = 0.01;
    let mut filter = BlockedBloomFilter::new(entries, fpr);

    // Insert strings, integers, or any type implementing Hash
    filter.insert_key("my_key");
    filter.insert_key(&12345);

    if filter.may_match_key("my_key") {
        println!("Key may be present");
    }
}
```

#### Using the Raw API (Power Users)
If you have your own hash (e.g., from a database or another hasher), you can skip the hashing step.

```rust
use bloomsday::BlockedBloomFilter;

fn main() {
    let mut filter = BlockedBloomFilter::new(1_000, 0.01);
    
    let my_hash: u64 = 0xDEADBEEF; // Pre-calculated hash
    filter.insert_hash(my_hash);

    if filter.may_match_hash(my_hash) {
        println!("Hash present");
    }
}
```

## Performance

Benchmarks run on `1,000,000` items with `0.01` false positive rate:

| Operation | Implementation | Time (avg) |
|-----------|----------------|------------|
| Negative Lookup | **bloomsday** | **~1.17 µs** |
| Negative Lookup | sbbf-rs | ~2.81 µs |
| Positive Lookup | **bloomsday** | **~1.26 µs** |
| Positive Lookup | sbbf-rs | ~2.93 µs |

Run benchmarks yourself with:
```bash
cargo bench
```

## License

MIT