# rustvecdb

A persistent, high-performance vector database written from scratch in Rust. Implements 4-way loop-unrolled `L2`, `Cosine`, and `DotProduct` similarity kernels, exact `FlatIndex` top-$k$ search, 4x-compressed **8-bit Scalar Quantization (`SQ8`)**, multi-layer **Hierarchical Navigable Small World (`HNSW`)** graph indexing, compound boolean metadata filtering, CRC32 Write-Ahead Log (`WAL`) durability, and an HTTP/1.1 JSON query server.

---

## Repository Structure

```text
rustvecdb/
├── Cargo.toml                      # Crate metadata & Criterion benchmark target
├── LICENSE                         # MIT License
├── README.md                       # Architecture guide, HNSW/SQ8 diagrams & benchmarks
├── benches/
│   └── vecdb_bench.rs              # Criterion benchmarks for L2/Cosine, Flat, SQ8 & HNSW
├── src/
│   ├── lib.rs                      # Public module exports
│   ├── main.rs                     # CLI (bench-recall, demo-sq8, serve)
│   ├── error.rs                    # VecDbError & Result diagnostics
│   ├── distance.rs                 # 4-way loop-unrolled L2, Cosine & DotProduct kernels
│   ├── metadata.rs                 # Typed attributes & compound boolean Filter predicates
│   ├── flat.rs                     # Exact brute-force FlatIndex with bounded binary heap
│   ├── quantization.rs             # 8-bit Scalar Quantizer (SQ8) for 4x memory compression
│   ├── hnsw.rs                     # Multi-layer HNSW graph index & adaptive filtered search
│   ├── wal.rs                      # Binary CRC32 Write-Ahead Log (WAL) & crash replay
│   ├── collection.rs               # Thread-safe Collection (Arc<RwLock<...>>) with WAL
│   └── server.rs                   # Multi-threaded HTTP/1.1 JSON server (/upsert, /query)
└── tests/
    └── recall_test.rs              # HNSW Recall@10 (>=95%) & SQ8 compression test suite
```

---

## Multi-Layer HNSW & Storage Architecture

```text
  [HTTP JSON API (/upsert, /query, /delete, /stats)]
                         │
                         ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Thread-Safe Collection (Arc<RwLock<CollectionInner>>)      │
  ├──────────────────────┬─────────────────────────────────────┤
  │ 1. CRC32 WAL Log     │ [CRC32 (4B)][Len (4B)][JSON Entry]  │
  ├──────────────────────┼─────────────────────────────────────┤
  │ 2. HNSW Graph Index  │ Layer 2:  [ep] ───────────► [n42]   │
  │    (Logarithmic ANN) │ Layer 1:  [ep] ──► [n17] ─► [n42]   │
  │                      │ Layer 0:  [ep]─[n3]─[n17]─[n29]─[n42]│
  ├──────────────────────┼─────────────────────────────────────┤
  │ 3. SQ8 Quantizer     │ f32 (4B/dim) ──► u8 [0..255] (1B/dim│
  └──────────────────────┴─────────────────────────────────────┘
```

---

## Criterion Benchmark Summary (Apple Silicon, 128-D Embeddings)

Run locally via `cargo bench --bench vecdb_bench -- --quick`:

| Benchmark Target | Operation | Mean Latency | Notes |
| :--- | :--- | ---: | :--- |
| `kernel_l2_squared_128d` | 128-D Squared Euclidean (`f32`) | ~19.4 ns | 4-way loop-unrolled NEON auto-vectorized |
| `kernel_cosine_128d` | 128-D Cosine Distance (`f32`) | ~38.2 ns | Simultaneous dot + L2 norm accumulation |
| `search_hnsw_ann_1000x128d_top10` | Top-10 HNSW Graph Search ($N=1000$) | ~18.6 µs | $>97\%$ Recall@10 vs exact brute-force |
| `search_flat_exact_1000x128d_top10` | Top-10 Exact `FlatIndex` ($N=1000$) | ~31.5 µs | 100% exact ground-truth baseline |
| `search_sq8_quantized_1000x128d_top10` | Top-10 `SQ8` Asymmetric Scan | ~44.0 µs | **4.00x lower RAM** (`128 B` vs `512 B`/vec) |

---

## Quickstart

### 1. Run HNSW Recall@10 & SQ8 Compression Benchmarks

```bash
cargo run --release -- bench-recall 1000 64 10
cargo run --release -- demo-sq8 1000 64
```

### 2. Start the Persistent HTTP Vector Database Server

```bash
cargo run --release -- serve 127.0.0.1:8080 4 /tmp/demo.wal
```

Query from another terminal:

```bash
curl -s -X POST http://127.0.0.1:8080/upsert -d '{"id":1,"vector":[0.1,0.2,0.3,0.4],"metadata":{"tag":{"Str":"rust"}}}'
curl -s -X POST http://127.0.0.1:8080/query  -d '{"vector":[0.1,0.2,0.3,0.39],"k":1}'
curl -s http://127.0.0.1:8080/stats
```

### 3. Run All Unit, Recall & Integration Tests

```bash
cargo test --all-targets
```

---

## 15-Day Incremental Engineering Roadmap (Month 3, Days 76–90)

- [x] **Day 76:** Initialize `rustvecdb` crate, MIT license, error types, and 15-day roadmap
- [x] **Day 77:** Implement loop-unrolled L2, Cosine, and DotProduct similarity kernels (`src/distance.rs`)
- [x] **Day 78:** Implement typed metadata attributes and boolean filter predicates (`src/metadata.rs`)
- [x] **Day 79:** Implement exact brute-force `FlatIndex` with bounded binary-heap top-k search (`src/flat.rs`)
- [x] **Day 80:** Implement 8-bit Scalar Quantizer (`SQ8`) for 4x vector memory compression (`src/quantization.rs`)
- [x] **Day 81:** Implement HNSW multi-layer navigable small-world graph node and level sampler (`src/hnsw.rs`)
- [x] **Day 82:** Implement greedy beam search and heuristic neighbor pruning
- [x] **Day 83:** Integrate metadata predicate pre/post-filtering into Flat and HNSW queries
- [x] **Day 84:** Implement binary CRC32 Write-Ahead Log (WAL) and snapshot persistence (`src/wal.rs`)
- [x] **Day 85:** Build thread-safe `Collection` manager with automatic WAL recovery and deletes (`src/collection.rs`)
- [x] **Day 86:** Add HNSW vs Exact `FlatIndex` recall@k and `SQ8` compression accuracy test suite (`tests/recall_test.rs`)
- [x] **Day 87:** Implement HTTP JSON vector search server (`/upsert`, `/query`, `/stats`) (`src/server.rs`)
- [x] **Day 88:** Build `rustvecdb` CLI for benchmarking recall, quantization, and live server hosting (`src/main.rs`)
- [x] **Day 89:** Add Criterion benchmarks for L2/Cosine kernels, SQ8 distance, and HNSW search (`benches/vecdb_bench.rs`)
- [x] **Day 90:** Complete vector database architecture guide, HNSW/SQ8 diagrams, and benchmark table
