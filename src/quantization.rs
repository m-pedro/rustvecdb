use std::collections::{BinaryHeap, HashMap};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecDbError};
use crate::flat::{HeapCandidate, SearchResult};
use crate::metadata::{Filter, Metadata};

/// Per-dimension affine quantization parameters mapping f32 <-> u8 [0..=255].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    pub dim: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // (max - min) / 255.0
}

impl ScalarQuantizer {
    /// Trains per-dimension min/max bounds across a sample of vectors.
    pub fn fit(dim: usize, vectors: &[Vec<f32>]) -> Result<Self> {
        if dim == 0 {
            return Err(VecDbError::EmptyVector);
        }
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        if vectors.is_empty() {
            return Ok(Self {
                dim,
                mins: vec![-1.0; dim],
                scales: vec![2.0 / 255.0; dim],
            });
        }

        for v in vectors {
            if v.len() != dim {
                return Err(VecDbError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
            for (d, &val) in v.iter().enumerate() {
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        let scales = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let span = mx - mn;
                if span <= f32::EPSILON {
                    1.0
                } else {
                    span / 255.0
                }
            })
            .collect();

        Ok(Self { dim, mins, scales })
    }

    /// Compresses an f32 vector (4 bytes/dim) into a u8 code vector (1 byte/dim).
    #[inline]
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.dim {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        let mut code = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let scaled = ((vector[d] - self.mins[d]) / self.scales[d]).round();
            code.push(scaled.clamp(0.0, 255.0) as u8);
        }
        Ok(code)
    }

    /// Reconstructs an approximate f32 vector from its u8 quantized codes.
    #[inline]
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        code.iter()
            .enumerate()
            .map(|(d, &q)| self.mins[d] + (q as f32) * self.scales[d])
            .collect()
    }

    /// Computes asymmetric squared L2 distance directly between an unquantized f32 query
    /// and a quantized u8 code vector without allocating a decoded Vec<f32>.
    #[inline]
    pub fn asymmetric_l2_squared(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for d in 0..self.dim {
            let reconstructed = self.mins[d] + (code[d] as f32) * self.scales[d];
            let diff = query[d] - reconstructed;
            sum += diff * diff;
        }
        sum
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizedRecord {
    id: u64,
    code: Vec<u8>,
    metadata: Metadata,
}

/// A 4x-compressed vector index storing 1-byte SQ8 codes instead of 4-byte f32 floats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedFlatIndex {
    quantizer: ScalarQuantizer,
    records: HashMap<u64, QuantizedRecord>,
}

impl QuantizedFlatIndex {
    pub fn new(quantizer: ScalarQuantizer) -> Self {
        Self {
            quantizer,
            records: HashMap::new(),
        }
    }

    pub fn upsert(&mut self, id: u64, vector: &[f32], metadata: Metadata) -> Result<()> {
        let code = self.quantizer.encode(vector)?;
        self.records.insert(
            id,
            QuantizedRecord {
                id,
                code,
                metadata,
            },
        );
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.quantizer.dim {
            return Err(VecDbError::DimensionMismatch {
                expected: self.quantizer.dim,
                actual: query.len(),
            });
        }
        if k == 0 || self.records.is_empty() {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<HeapCandidate> = BinaryHeap::with_capacity(k + 1);
        for rec in self.records.values() {
            if let Some(f) = filter {
                if !f.matches(&rec.metadata) {
                    continue;
                }
            }
            let dist = self.quantizer.asymmetric_l2_squared(query, &rec.code);
            if heap.len() < k {
                heap.push(HeapCandidate {
                    id: rec.id,
                    distance: dist,
                });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.distance {
                    heap.pop();
                    heap.push(HeapCandidate {
                        id: rec.id,
                        distance: dist,
                    });
                }
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| {
                let rec = &self.records[&c.id];
                SearchResult {
                    id: rec.id,
                    distance: c.distance,
                    metadata: rec.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id)));
        Ok(results)
    }

    /// Returns (compressed_bytes, equivalent_f32_bytes) to measure the 4x compression ratio.
    pub fn memory_footprint_bytes(&self) -> (usize, usize) {
        let compressed = self.records.len() * self.quantizer.dim;
        let raw_f32 = compressed * std::mem::size_of::<f32>();
        (compressed, raw_f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_quantization_preserves_nearest_neighbors_with_4x_compression() {
        let sample = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![5.0, 5.0, 5.0, 5.0],
            vec![10.0, 10.0, 10.0, 10.0],
        ];
        let sq = ScalarQuantizer::fit(4, &sample).unwrap();
        let mut qidx = QuantizedFlatIndex::new(sq);

        for (i, v) in sample.iter().enumerate() {
            qidx.upsert((i + 1) as u64, v, Metadata::new()).unwrap();
        }

        let (compressed, raw) = qidx.memory_footprint_bytes();
        assert_eq!(raw, compressed * 4);

        let hits = qidx.search(&[0.9, 0.9, 0.9, 0.9], 2, None).unwrap();
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[1].id, 1);
    }
}
