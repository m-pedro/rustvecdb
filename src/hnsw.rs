use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Result, VecDbError};
use crate::metadata::Metadata;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
    /// Maximum number of bidirectional links per node on layers >= 1.
    pub m: usize,
    /// Maximum number of bidirectional links per node on layer 0 (typically 2 * M).
    pub m_max0: usize,
    /// Candidate beam width during graph construction.
    pub ef_construction: usize,
    /// Default candidate beam width during query search.
    pub ef_search: usize,
    /// Level normalization multiplier: 1.0 / ln(M).
    pub level_mult: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16usize;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 64,
            ef_search: 32,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
    pub level: usize,
    /// Adjacency lists per layer: `links[layer]` holds neighbor node IDs at `layer`.
    pub links: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) params: HnswParams,
    pub(crate) nodes: HashMap<u64, HnswNode>,
    pub(crate) entry_point: Option<u64>,
    pub(crate) max_layer: usize,
    pub(crate) rng_state: u64,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self> {
        if dim == 0 {
            return Err(VecDbError::EmptyVector);
        }
        if params.m == 0 || params.ef_construction == 0 {
            return Err(VecDbError::InvalidParameter(
                "HNSW M and ef_construction must be > 0".into(),
            ));
        }
        Ok(Self {
            dim,
            metric,
            params,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Samples a random layer level using SplitMix64 and the HNSW exponential decay distribution:
    /// `level = floor(-ln(uniform(0, 1)) * level_mult)`.
    pub fn sample_level(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;

        // Convert top 53 bits to uniform f64 in (0, 1]
        let u = ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let lvl = (-u.ln() * self.params.level_mult).floor() as usize;
        lvl.min(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_geometric_level_sampler_distribution() {
        let mut idx = HnswIndex::new(8, Metric::L2, HnswParams::default()).unwrap();
        let mut counts = [0usize; 17];
        for _ in 0..2000 {
            let l = idx.sample_level();
            counts[l] += 1;
        }
        // Layer 0 must contain the vast majority of nodes, with monotonically fewer nodes at higher layers
        assert!(counts[0] > 1700);
        assert!(counts[1] > 0);
        assert!(counts[0] > counts[1]);
    }
}
