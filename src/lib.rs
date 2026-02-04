use std::f64::consts::LN_2;
use std::hash::{Hash, Hasher};
use xxhash_rust::xxh64::Xxh64;

/// A cache-line blocked Bloom filter.
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

    /// Creates a new filter with the given entries and false positive rate.
    pub fn new(entries: usize, fpr: f64) -> Self {
        Self::new_with_seed(entries, fpr, 0)
    }

    /// Creates a new filter with a custom seed.
    pub fn new_with_seed(entries: usize, fpr: f64, seed: u64) -> Self {
        let bits_per_key = Self::bloom_bits_per_key(fpr);
        let mut num_blocks = ((entries * bits_per_key + 255) / 256) as u32;
        if num_blocks == 0 {
            num_blocks = 1;
        }
        
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        unsafe {
            blocks.set_len(num_blocks as usize);
            std::ptr::write_bytes(blocks.as_mut_ptr(), 0, num_blocks as usize);
        }
        Self { blocks, num_blocks, seed }
    }

    #[inline(always)]
    fn fast_map(&self, hash: u32) -> usize {
        ((hash as u64 * self.num_blocks as u64) >> 32) as usize
    }

    /// Inserts a hash into the filter.
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

    /// Checks if the filter might contain the hash.
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

    /// Hashes the key and inserts it.
    #[inline]
    pub fn insert_key<T: Hash + ?Sized>(&mut self, key: &T) {
        let mut hasher = Xxh64::new(self.seed);
        key.hash(&mut hasher);
        self.insert_hash(hasher.finish());
    }

    /// Hashes the key and checks if it might be present.
    #[inline]
    pub fn may_match_key<T: Hash + ?Sized>(&self, key: &T) -> bool {
        let mut hasher = Xxh64::new(self.seed);
        key.hash(&mut hasher);
        self.may_match_hash(hasher.finish())
    }

    fn bloom_bits_per_key(fpr: f64) -> usize {
        if fpr <= 0.0 || fpr >= 1.0 {
            return 10;
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
            assert!(bf.may_match_hash(*h));
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
        assert!(actual_fpr < fpr * 2.5);
    }

    #[test]
    fn test_zero_entries_init() {
        let mut bf = BlockedBloomFilter::new(0, 0.01);
        assert!(bf.num_blocks >= 1);
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
        assert!(bf1.may_match_key(key));
        assert!(!bf2.may_match_key(key));
    }

    #[test]
    fn test_saturation() {
        let mut bf = BlockedBloomFilter::new(10, 0.01);
        for i in 0..1000 {
            bf.insert_hash(i as u64);
        }
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