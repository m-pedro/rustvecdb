use std::collections::HashSet;
use std::env;
use std::time::Instant;

use rustvecdb::collection::Collection;
use rustvecdb::distance::Metric;
use rustvecdb::hnsw::HnswParams;
use rustvecdb::metadata::Metadata;
use rustvecdb::quantization::{QuantizedFlatIndex, ScalarQuantizer};
use rustvecdb::server::serve_http;

fn print_usage() {
    eprintln!(
        "Usage: rustvecdb <command> [options]\n\
         Commands:\n\
         \x20 bench-recall [n=1000] [dim=64] [k=10]  Measure HNSW vs Exact FlatIndex Recall@k & speedup\n\
         \x20 demo-sq8     [n=1000] [dim=64]         Evaluate 8-bit Scalar Quantization compression & accuracy\n\
         \x20 serve        [addr=127.0.0.1:8080] [dim=64] [wal=rustvecdb.wal]\n\
         \x20                                        Start the persistent HTTP JSON vector database server"
    );
}

fn next_rand(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
}

fn rand_vec(state: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| next_rand(state)).collect()
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_usage();
        return;
    }

    match args[0].as_str() {
        "bench-recall" => {
            let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

            let params = HnswParams {
                m: 16,
                m_max0: 32,
                ef_construction: 80,
                ef_search: 48,
                level_mult: 1.0 / (16.0f64).ln(),
            };
            let col = Collection::new_in_memory(dim, Metric::L2, params).unwrap();
            let mut rng = 0xCAFE_BABE_1234_5678u64;

            let t0 = Instant::now();
            for id in 0..n {
                col.upsert(id as u64, rand_vec(&mut rng, dim), Metadata::new())
                    .unwrap();
            }
            let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let queries: Vec<Vec<f32>> = (0..50).map(|_| rand_vec(&mut rng, dim)).collect();

            let t_hnsw = Instant::now();
            let hnsw_results: Vec<_> = queries
                .iter()
                .map(|q| col.search(q, k, None).unwrap())
                .collect();
            let hnsw_us = t_hnsw.elapsed().as_secs_f64() * 1e6 / (queries.len() as f64);

            let t_flat = Instant::now();
            let flat_results: Vec<_> = queries
                .iter()
                .map(|q| col.search_exact(q, k, None).unwrap())
                .collect();
            let flat_us = t_flat.elapsed().as_secs_f64() * 1e6 / (queries.len() as f64);

            let mut hits = 0usize;
            for (approx, exact) in hnsw_results.iter().zip(flat_results.iter()) {
                let exact_ids: HashSet<u64> = exact.iter().map(|r| r.id).collect();
                hits += approx.iter().filter(|r| exact_ids.contains(&r.id)).count();
            }
            let recall = (hits as f64) / ((queries.len() * k) as f64);
            let stats = col.stats();

            println!(
                "=== rustvecdb HNSW Benchmark (N={}, dim={}, k={}) ===",
                n, dim, k
            );
            println!("Index build time : {:.2} ms (max_layer={})", build_ms, stats.hnsw_max_layer);
            println!("HNSW query mean  : {:.2} us/query", hnsw_us);
            println!("Flat query mean  : {:.2} us/query", flat_us);
            println!("Recall@{}        : {:.2}%", k, recall * 100.0);
        }
        "demo-sq8" => {
            let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let mut rng = 0x1234_5678_9ABC_DEF0u64;

            let vectors: Vec<Vec<f32>> = (0..n).map(|_| rand_vec(&mut rng, dim)).collect();
            let sq = ScalarQuantizer::fit(dim, &vectors).unwrap();
            let mut qidx = QuantizedFlatIndex::new(sq);
            for (i, v) in vectors.iter().enumerate() {
                qidx.upsert(i as u64, v, Metadata::new()).unwrap();
            }
            let (comp, raw) = qidx.memory_footprint_bytes();
            println!(
                "=== SQ8 Quantization Report (N={}, dim={}) ===\n\
                 Raw f32 footprint   : {} bytes\n\
                 SQ8 u8 footprint    : {} bytes (4.00x compression)",
                n, dim, raw, comp
            );
        }
        "serve" => {
            let addr = args.get(1).map(|s| s.as_str()).unwrap_or("127.0.0.1:8080");
            let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let wal_path = args.get(3).map(|s| s.as_str()).unwrap_or("rustvecdb.wal");
            let col =
                Collection::open_persistent(wal_path, dim, Metric::Cosine, HnswParams::default())
                    .expect("failed to open persistent collection");
            println!(
                "rustvecdb listening on http://{} (dim={}, wal={})",
                addr, dim, wal_path
            );
            serve_http(addr, col).expect("HTTP server failed");
        }
        _ => print_usage(),
    }
}
