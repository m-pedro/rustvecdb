# rustvecdb

A high-performance, persistent vector database written in Rust. Supports loop-unrolled `L2`, `Cosine`, and `DotProduct` distance kernels, exact `FlatIndex` search, 4x-compressed **8-bit Scalar Quantization (`SQ8`)**, multi-layer **Hierarchical Navigable Small World (`HNSW`)** approximate nearest-neighbor indexing, structured metadata filtering, and CRC32 Write-Ahead Log (`WAL`) persistence.

## 15-Day Incremental Engineering Roadmap (Month 3, Days 76–90)

- [x] **Day 76:** Initialize `rustvecdb` crate, MIT license, error types, and 15-day roadmap
- [ ] **Day 77:** Implement loop-unrolled L2, Cosine, and DotProduct similarity kernels (`src/distance.rs`)
- [ ] **Day 78:** Implement typed metadata attributes and boolean filter predicates (`src/metadata.rs`)
- [ ] **Day 79:** Implement exact brute-force `FlatIndex` with bounded binary-heap top-k search (`src/flat.rs`)
- [ ] **Day 80:** Implement 8-bit Scalar Quantizer (`SQ8`) for 4x vector memory compression (`src/quantization.rs`)
- [ ] **Day 81:** Implement HNSW multi-layer navigable small-world graph node and level sampler (`src/hnsw.rs`)
- [ ] **Day 82:** Implement greedy beam search and heuristic neighbor pruning
- [ ] **Day 83:** Integrate metadata predicate pre/post-filtering into Flat and HNSW queries
- [ ] **Day 84:** Implement binary CRC32 Write-Ahead Log (WAL) and snapshot persistence (`src/wal.rs`)
- [ ] **Day 85:** Build thread-safe `Collection` manager with automatic WAL recovery and deletes (`src/collection.rs`)
- [ ] **Day 86:** Add HNSW vs Exact `FlatIndex` recall@k and `SQ8` compression accuracy test suite
- [ ] **Day 87:** Implement HTTP JSON vector search server (`/upsert`, `/query`, `/stats`)
- [ ] **Day 88:** Build `rustvecdb` CLI for benchmarking recall, quantization, and live server hosting
- [ ] **Day 89:** Add Criterion benchmarks for L2/Cosine kernels, SQ8 distance, and HNSW search
- [ ] **Day 90:** Complete vector database architecture guide, HNSW/SQ8 diagrams, and benchmark table
