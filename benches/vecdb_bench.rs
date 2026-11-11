use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustvecdb::distance::{cosine_distance, l2_squared, Metric};
use rustvecdb::flat::FlatIndex;
use rustvecdb::hnsw::{HnswIndex, HnswParams};
use rustvecdb::metadata::Metadata;
use rustvecdb::quantization::{QuantizedFlatIndex, ScalarQuantizer};

fn make_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..dim)
        .map(|_| {
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9).wrapping_add(1);
            ((s >> 40) as f32) / ((1u64 << 24) as f32)
        })
        .collect()
}

fn bench_distance_and_indexes(c: &mut Criterion) {
    let dim = 128;
    let a = make_vec(1, dim);
    let b = make_vec(2, dim);

    c.bench_function("kernel_l2_squared_128d", |bencher| {
        bencher.iter(|| black_box(l2_squared(black_box(&a), black_box(&b))))
    });

    c.bench_function("kernel_cosine_128d", |bencher| {
        bencher.iter(|| black_box(cosine_distance(black_box(&a), black_box(&b))))
    });

    let n = 1000usize;
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vec(i as u64, dim)).collect();
    let query = make_vec(9999, dim);

    let mut flat = FlatIndex::new(dim, Metric::L2).unwrap();
    let mut hnsw = HnswIndex::new(dim, Metric::L2, HnswParams::default()).unwrap();
    let sq = ScalarQuantizer::fit(dim, &vectors).unwrap();
    let mut qflat = QuantizedFlatIndex::new(sq);

    for (i, v) in vectors.iter().enumerate() {
        flat.upsert(i as u64, v.clone(), Metadata::new()).unwrap();
        hnsw.upsert(i as u64, v.clone(), Metadata::new()).unwrap();
        qflat.upsert(i as u64, v, Metadata::new()).unwrap();
    }

    c.bench_function("search_flat_exact_1000x128d_top10", |bencher| {
        bencher.iter(|| black_box(flat.search(black_box(&query), 10, None).unwrap()))
    });

    c.bench_function("search_sq8_quantized_1000x128d_top10", |bencher| {
        bencher.iter(|| black_box(qflat.search(black_box(&query), 10, None).unwrap()))
    });

    c.bench_function("search_hnsw_ann_1000x128d_top10", |bencher| {
        bencher.iter(|| black_box(hnsw.search(black_box(&query), 10).unwrap()))
    });
}

criterion_group!(benches, bench_distance_and_indexes);
criterion_main!(benches);
