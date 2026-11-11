use std::collections::HashSet;

use rustvecdb::collection::Collection;
use rustvecdb::distance::Metric;
use rustvecdb::hnsw::HnswParams;
use rustvecdb::metadata::Metadata;
use rustvecdb::quantization::{QuantizedFlatIndex, ScalarQuantizer};

/// Deterministic SplitMix64 pseudo-random generator for reproducible synthetic embeddings.
struct DetRng(u64);

impl DetRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn next_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

#[test]
fn test_hnsw_vs_flat_recall_at_10_exceeds_95_percent() {
    let dim = 32;
    let n_vectors = 600;
    let n_queries = 30;
    let k = 10;

    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 80,
        ef_search: 48,
        level_mult: 1.0 / (16.0f64).ln(),
    };

    let col = Collection::new_in_memory(dim, Metric::L2, params).unwrap();
    let mut rng = DetRng(42);

    for id in 0..n_vectors {
        col.upsert(id as u64, rng.next_vec(dim), Metadata::new())
            .unwrap();
    }

    let mut total_hits = 0usize;
    for _ in 0..n_queries {
        let q = rng.next_vec(dim);
        let approx = col.search(&q, k, None).unwrap();
        let exact = col.search_exact(&q, k, None).unwrap();

        let exact_ids: HashSet<u64> = exact.into_iter().map(|r| r.id).collect();
        total_hits += approx.iter().filter(|r| exact_ids.contains(&r.id)).count();
    }

    let recall = (total_hits as f64) / ((n_queries * k) as f64);
    assert!(
        recall >= 0.95,
        "expected HNSW Recall@10 >= 0.95, got {:.3}",
        recall
    );
}

#[test]
fn test_sq8_quantization_recall_and_compression_ratio() {
    let dim = 32;
    let n_vectors = 400;
    let k = 10;
    let mut rng = DetRng(1337);

    let vectors: Vec<Vec<f32>> = (0..n_vectors).map(|_| rng.next_vec(dim)).collect();
    let sq = ScalarQuantizer::fit(dim, &vectors).unwrap();
    let mut q_idx = QuantizedFlatIndex::new(sq);
    let col = Collection::new_in_memory(dim, Metric::L2, HnswParams::default()).unwrap();

    for (i, v) in vectors.iter().enumerate() {
        q_idx.upsert(i as u64, v, Metadata::new()).unwrap();
        col.upsert(i as u64, v.clone(), Metadata::new()).unwrap();
    }

    let (compressed_bytes, raw_bytes) = q_idx.memory_footprint_bytes();
    assert_eq!(raw_bytes, compressed_bytes * 4);

    let mut matches = 0usize;
    let n_queries = 20;
    for _ in 0..n_queries {
        let q = rng.next_vec(dim);
        let sq_res = q_idx.search(&q, k, None).unwrap();
        let exact_res = col.search_exact(&q, k, None).unwrap();
        let exact_ids: HashSet<u64> = exact_res.into_iter().map(|r| r.id).collect();
        matches += sq_res.iter().filter(|r| exact_ids.contains(&r.id)).count();
    }

    let sq_recall = (matches as f64) / ((n_queries * k) as f64);
    assert!(
        sq_recall >= 0.90,
        "expected SQ8 Recall@10 >= 0.90, got {:.3}",
        sq_recall
    );
}
