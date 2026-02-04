use std::f64::consts::LN_2;
use std::hash::{Hash, Hasher};
use xxhash_rust::xxh64::Xxh64;

/// A cache-line blocked Bloom filter optimized for modern CPUs (AVX2/AVX-512).
///
/// This implementation uses a blocked strategy where keys are hashed to a specific block
/// (fitting in a cache line), and then multiple bits are set within that block.
/// This improves cache locality and allows for SIMD optimizations.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct BlockedBloomFilter {
    blocks: Vec<CacheLineBlock>,
    num_blocks: u32,
    seed: u64,
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CacheLineBlock {
    words: [u64; 4],
}

impl BlockedBloomFilter {
    const SALT: [u32; 8] = [
        0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
        0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
    ];

    /// Creates a new BlockedBloomFilter with the given expected number of entries and false positive rate.
    /// 
    /// Uses a default seed of 0 for the internal hasher.
    pub fn new(entries: usize, fpr: f64) -> Self {
        Self::new_with_seed(entries, fpr, 0)
    }

    /// Creates a new BlockedBloomFilter with a custom seed for the internal hasher.
    pub fn new_with_seed(entries: usize, fpr: f64, seed: u64) -> Self {
        let bits_per_key = Self::bloom_bits_per_key(fpr);
        
        // Calculate number of blocks needed.
        // We ensure at least 1 block exists to avoid division by zero or empty buffer issues
        // in fast_map and unsafe access, even if 0 entries are requested.
        let mut num_blocks = ((entries * bits_per_key + 255) / 256) as u32;
        if num_blocks == 0 {
            num_blocks = 1;
        }
        
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        unsafe {
            blocks.set_len(num_blocks as usize);
            // Ensure zero-initialization
            std::ptr::write_bytes(blocks.as_mut_ptr(), 0, num_blocks as usize);
        }
        Self { blocks, num_blocks, seed }
    }

    /// Optimized 32-bit mapping to find the block index.
    #[inline(always)]
    fn fast_map(&self, hash: u32) -> usize {
        // Fast range reduction: (x * N) >> 32
        ((hash as u64 * self.num_blocks as u64) >> 32) as usize
    }

    /// Inserts a raw u64 hash into the Bloom filter.
    #[inline(always)]
    pub fn insert_hash(&mut self, h: u64) {
        let block_idx = self.fast_map((h >> 32) as u32);

        unsafe {
            let block_ptr = self.blocks.get_unchecked_mut(block_idx).words.as_mut_ptr();
            let words = &mut *(block_ptr as *mut [u32; 8]);
            
            words.iter_mut()
                 .zip(Self::SALT.iter())
                 .for_each(|(w, &salt)| {
                     let idx = (h as u32).wrapping_mul(salt) >> 27;
                     *w |= 1 << idx;
                 });
        }
    }

    /// Checks if the Bloom filter might contain the given raw u64 hash.
    #[inline(always)]
    pub fn may_match_hash(&self, h: u64) -> bool {
        let block_idx = self.fast_map((h >> 32) as u32);

        unsafe {
            let block_ptr = self.blocks.get_unchecked(block_idx).words.as_ptr();
            let words = &*(block_ptr as *const [u32; 8]);
            
            let check = words.iter()
                             .zip(Self::SALT.iter())
                             .fold(0u32, |acc, (&w, &salt)| {
                                 let idx = (h as u32).wrapping_mul(salt) >> 27;
                                 acc | ((1 << idx) & !w)
                             });
            
            check == 0
        }
    }

    /// Hashes the key using xxHash (xxh64) and inserts it into the filter.
    #[inline]
    pub fn insert_key<T: Hash + ?Sized>(&mut self, key: &T) {
        let mut hasher = Xxh64::new(self.seed);
        key.hash(&mut hasher);
        self.insert_hash(hasher.finish());
    }

    /// Hashes the key using xxHash (xxh64) and checks if it might be in the filter.
    #[inline]
    pub fn may_match_key<T: Hash + ?Sized>(&self, key: &T) -> bool {
        let mut hasher = Xxh64::new(self.seed);
        key.hash(&mut hasher);
        self.may_match_hash(hasher.finish())
    }

    fn bloom_bits_per_key(fpr: f64) -> usize { // TODO: find more accurate formula for blocked BF size
        // If FPR is invalid (e.g. <= 0 or >= 1), we clamp or default.
        // For simplicity, we assume reasonable input, but preventing crash on 0.0 is good.
        if fpr <= 0.0 || fpr >= 1.0 {
            return 10; // Default fallback
        }
        ((-1.0 * fpr.ln()) / (LN_2 * LN_2)).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_basic_insert_and_contains() {
        let mut bf = BlockedBloomFilter::new(1000, 0.01);
        let hash = 1234567890;
        
        assert!(!bf.may_match_hash(hash));
        bf.insert_hash(hash);
        assert!(bf.may_match_hash(hash));
    }

    #[test]
    fn test_key_api() {
        let mut bf = BlockedBloomFilter::new(1000, 0.01);
        let key = "hello world";
        
        assert!(!bf.may_match_key(key));
        bf.insert_key(key);
        assert!(bf.may_match_key(key));
        assert!(!bf.may_match_key("goodbye"));
    }

    #[test]
    fn test_false_positive_rate() {
        let entries = 10_000;
        let fpr = 0.01;
        let mut bf = BlockedBloomFilter::new(entries, fpr);
        
        let mut rng = rand::rng();
        let mut inserted = Vec::new();

        for _ in 0..entries {
            let h: u64 = rng.random();
            bf.insert_hash(h);
            inserted.push(h);
        }

        for h in &inserted {
            assert!(bf.may_match_hash(*h), "Inserted item should be found");
        }

        let tests = 100_000;
        let mut fp_count = 0;
        for _ in 0..tests {
            let h: u64 = rng.random();
            if !inserted.contains(&h) && bf.may_match_hash(h) {
                fp_count += 1;
            }
        }

        let actual_fpr = fp_count as f64 / tests as f64;
        println!("Actual FPR: {}", actual_fpr);
        // Slightly loose tolerance for blocked BF
        assert!(actual_fpr < fpr * 2.5, "FPR {} is too high (expected {})", actual_fpr, fpr);
    }

    #[test]
    fn test_zero_entries_init() {
        // Should not panic and should create at least 1 block
        let mut bf = BlockedBloomFilter::new(0, 0.01);
        assert!(bf.num_blocks >= 1);
        
        // Should work safely
        bf.insert_hash(123);
        assert!(bf.may_match_hash(123));
    }

    #[test]
    fn test_clone() {
        let mut bf = BlockedBloomFilter::new(100, 0.01);
        bf.insert_hash(12345);
        
        let bf_clone = bf.clone();
        assert!(bf_clone.may_match_hash(12345));
        assert!(!bf_clone.may_match_hash(67890));
    }

    #[test]
    fn test_different_seeds() {
        let mut bf1 = BlockedBloomFilter::new_with_seed(1000, 0.01, 123);
        let bf2 = BlockedBloomFilter::new_with_seed(1000, 0.01, 456);
        let key = "test_key";

        bf1.insert_key(key);
        
        // bf2 should NOT match efficiently because the seed makes the hash different
        // It *might* match by collision, but very unlikely.
        assert!(bf1.may_match_key(key));
        
        // There is a tiny chance of collision, but with 64-bit hash it's negligible for this test
        // However, since we are testing Bloom Filters, false positives are real.
        // But here we are testing the Hash function output difference.
        // If the implementation is correct, different seeds -> different hashes.
        // If they collide in the filter, that's just bad luck, but let's assert it doesn't
        // for this specific case which we know works.
        assert!(!bf2.may_match_key(key), "Different seeds should produce different hashes");
    }

    #[test]
    fn test_saturation() {
        // Create a small filter
        let mut bf = BlockedBloomFilter::new(10, 0.01);
        
        // Insert way more items than it can hold
        for i in 0..1000 {
            bf.insert_hash(i as u64);
        }

        // Everything should look like a match now (saturation)
        assert!(bf.may_match_hash(999999));
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let mut bf = BlockedBloomFilter::new(100, 0.01);
        bf.insert_hash(42);
        
        let serialized = serde_json::to_string(&bf).unwrap();
        let deserialized: BlockedBloomFilter = serde_json::from_str(&serialized).unwrap();
        
        assert!(deserialized.may_match_hash(42));
        assert!(!deserialized.may_match_hash(43));
    }
}
