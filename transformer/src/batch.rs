/// Sequence length batching for efficient training.
///
/// Paper 5.1: "Our approach batches sentences of similar length together
/// to minimize padding overhead."
use rand::seq::SliceRandom;
use rand::thread_rng;

/// A bucket-based batcher that groups sequences by approximate length.
///
/// Sequences are sorted into buckets by length (with configurable bucket width),
/// then shuffled within each bucket to produce training batches.
pub struct BucketBatcher {
    /// Bucket width in tokens (default: 8)
    pub bucket_width: usize,
    /// Batch size (number of sequences per batch)
    pub batch_size: usize,
    /// Pairs of (src, tgt) sequences
    pub data: Vec<(Vec<usize>, Vec<usize>)>,
    /// Indices for each bucket
    buckets: Vec<Vec<usize>>,
    /// RNG for shuffling
    rng: rand::rngs::ThreadRng,
}

impl BucketBatcher {
    /// Create a new bucket batcher.
    ///
    /// - `data`: Vec of (source_tokens, target_tokens) pairs
    /// - `batch_size`: sequences per batch
    /// - `bucket_width`: token length bucket size (default 8)
    pub fn new(
        data: Vec<(Vec<usize>, Vec<usize>)>,
        batch_size: usize,
        bucket_width: usize,
    ) -> Self {
        let rng = thread_rng();
        let mut batcher = Self {
            bucket_width,
            batch_size,
            data,
            buckets: Vec::new(),
            rng,
        };
        batcher.build_buckets();
        batcher
    }

    /// Build buckets based on sequence length.
    fn build_buckets(&mut self) {
        // Determine number of buckets based on max sequence length
        let max_len = self
            .data
            .iter()
            .map(|(src, tgt)| src.len().max(tgt.len()))
            .max()
            .unwrap_or(0);
        let n_buckets = if self.bucket_width == 0 {
            1
        } else {
            (max_len / self.bucket_width) + 1
        };
        self.buckets = vec![Vec::new(); n_buckets];

        for (i, (src, tgt)) in self.data.iter().enumerate() {
            let len = src.len().max(tgt.len());
            let bucket_idx = (len / self.bucket_width).min(n_buckets - 1);
            self.buckets[bucket_idx].push(i);
        }
    }

    /// Return the total number of batches.
    pub fn num_batches(&self) -> usize {
        let total: usize = self.buckets.iter().map(|b| b.len()).sum();
        (total + self.batch_size - 1) / self.batch_size
    }

    /// Get all batches, shuffled within buckets and across buckets.
    ///
    /// Each batch is a Vec of (src_tokens, tgt_tokens) pairs.
    pub fn batches(&mut self) -> Vec<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut all_indices = Vec::new();

        // Shuffle within each bucket, then flatten
        for bucket in &self.buckets {
            let mut shuffled = bucket.clone();
            shuffled.shuffle(&mut self.rng);
            all_indices.extend(shuffled);
        }

        // Shuffle inter-bucket order as well
        all_indices.shuffle(&mut self.rng);

        // Create batches
        let mut result = Vec::new();
        for chunk in all_indices.chunks(self.batch_size) {
            let batch: Vec<(Vec<usize>, Vec<usize>)> = chunk
                .iter()
                .map(|&idx| {
                    let (src, tgt) = &self.data[idx];
                    (src.clone(), tgt.clone())
                })
                .collect();
            result.push(batch);
        }

        result
    }

    /// Print bucket statistics for debugging.
    pub fn print_stats(&self) {
        println!("  BucketBatcher stats:");
        println!("    Total sequences: {}", self.data.len());
        println!("    Bucket width: {}", self.bucket_width);
        println!("    Number of buckets: {}", self.buckets.len());
        println!("    Batch size: {}", self.batch_size);
        println!("    Total batches: {}", self.num_batches());
        for (i, bucket) in self.buckets.iter().enumerate() {
            if !bucket.is_empty() {
                let min_len = i * self.bucket_width;
                let max_len = min_len + self.bucket_width;
                println!(
                    "    Bucket {} (len {}-{}): {} sequences",
                    i,
                    min_len,
                    max_len,
                    bucket.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_batcher_basic() {
        let data: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![1, 2, 3], vec![4, 5, 6]),
            (vec![7, 8], vec![9, 10]),
            (vec![11, 12, 13, 14], vec![15, 16, 17, 18]),
            (vec![19], vec![20]),
        ];
        let mut batcher = BucketBatcher::new(data, 2, 4);
        assert!(batcher.num_batches() >= 2);
        let batches = batcher.batches();
        assert!(!batches.is_empty());
        // Each batch should have at most batch_size items
        for batch in &batches {
            assert!(batch.len() <= 2);
        }
    }

    #[test]
    fn test_bucket_batcher_all_data() {
        let data: Vec<(Vec<usize>, Vec<usize>)> = (0..10)
            .map(|i| {
                let len = (i % 5) + 3;
                (vec![1; len], vec![2; len])
            })
            .collect();
        let total = data.len();
        let mut batcher = BucketBatcher::new(data, 4, 4);
        let batches = batcher.batches();
        let batched_count: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(batched_count, total);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<(Vec<usize>, Vec<usize>)> = vec![];
        let mut batcher = BucketBatcher::new(data, 4, 8);
        assert_eq!(batcher.num_batches(), 0);
        let batches = batcher.batches();
        assert!(batches.is_empty());
    }
}

