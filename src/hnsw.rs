use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Result, VecDbError};
use crate::flat::{HeapCandidate, SearchResult};
use crate::metadata::{Filter, Metadata};

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

        let u = ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let lvl = (-u.ln() * self.params.level_mult).floor() as usize;
        lvl.min(16)
    }

    /// Inserts or updates a vector in the multi-layer HNSW graph.
    pub fn upsert(&mut self, id: u64, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }

        if self.nodes.contains_key(&id) {
            self.remove(id);
        }

        let level = self.sample_level();
        let new_node = HnswNode {
            id,
            vector: vector.clone(),
            metadata,
            level,
            links: vec![Vec::new(); level + 1],
        };

        let Some(mut curr_ep) = self.entry_point else {
            self.max_layer = level;
            self.entry_point = Some(id);
            self.nodes.insert(id, new_node);
            return Ok(());
        };

        let curr_max_layer = self.max_layer;

        // Phase 1: Greedy 1-NN descent from top layer down to `level + 1`
        if curr_max_layer > level {
            for lc in ((level + 1)..=curr_max_layer).rev() {
                curr_ep = self.search_layer_greedy(&vector, curr_ep, lc);
            }
        }

        self.nodes.insert(id, new_node);

        // Phase 2: Beam search with `ef_construction` from `min(level, curr_max_layer)` down to 0
        let top_connect_layer = level.min(curr_max_layer);
        for lc in (0..=top_connect_layer).rev() {
            let candidates = self.search_layer(&vector, curr_ep, self.params.ef_construction, lc);
            let m_limit = if lc == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };

            let selected: Vec<u64> = candidates
                .iter()
                .filter(|c| c.id != id)
                .take(m_limit)
                .map(|c| c.id)
                .collect();

            if let Some(first) = selected.first() {
                curr_ep = *first;
            }

            if let Some(node) = self.nodes.get_mut(&id) {
                node.links[lc] = selected.clone();
            }

            // Add bidirectional edges and prune neighbor lists exceeding `m_limit`
            for nbr_id in selected {
                let maybe_prune = if let Some(nbr) = self.nodes.get_mut(&nbr_id) {
                    if lc < nbr.links.len() && !nbr.links[lc].contains(&id) {
                        nbr.links[lc].push(id);
                        if nbr.links[lc].len() > m_limit {
                            Some((nbr.vector.clone(), nbr.links[lc].clone()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((nbr_vec, link_ids)) = maybe_prune {
                    let mut scored: Vec<(u64, f32)> = link_ids
                        .into_iter()
                        .filter_map(|target_id| {
                            self.nodes
                                .get(&target_id)
                                .map(|t| (target_id, self.metric.distance(&nbr_vec, &t.vector)))
                        })
                        .collect();
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    scored.truncate(m_limit);
                    if let Some(nbr) = self.nodes.get_mut(&nbr_id) {
                        nbr.links[lc] = scored.into_iter().map(|(nid, _)| nid).collect();
                    }
                }
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }

        Ok(())
    }

    /// Removes a node from the HNSW graph and cleans up incident neighbor links.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(removed) = self.nodes.remove(&id) else {
            return false;
        };

        for (lc, nbrs) in removed.links.iter().enumerate() {
            for &nbr_id in nbrs {
                if let Some(nbr) = self.nodes.get_mut(&nbr_id) {
                    if lc < nbr.links.len() {
                        nbr.links[lc].retain(|&x| x != id);
                    }
                }
            }
        }

        if self.entry_point == Some(id) {
            if let Some((&new_ep, node)) = self.nodes.iter().max_by_key(|(_, n)| n.level) {
                self.entry_point = Some(new_ep);
                self.max_layer = node.level;
            } else {
                self.entry_point = None;
                self.max_layer = 0;
            }
        }
        true
    }

    /// Greedy 1-nearest-neighbor walk on `layer` starting from `entry_id`.
    pub(crate) fn search_layer_greedy(&self, query: &[f32], entry_id: u64, layer: usize) -> u64 {
        let mut curr_id = entry_id;
        let Some(curr_node) = self.nodes.get(&curr_id) else {
            return entry_id;
        };
        let mut curr_dist = self.metric.distance(query, &curr_node.vector);

        loop {
            let mut improved = false;
            if let Some(node) = self.nodes.get(&curr_id) {
                if let Some(neighbors) = node.links.get(layer) {
                    for &nbr_id in neighbors {
                        if let Some(nbr_node) = self.nodes.get(&nbr_id) {
                            let d = self.metric.distance(query, &nbr_node.vector);
                            if d < curr_dist {
                                curr_dist = d;
                                curr_id = nbr_id;
                                improved = true;
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        curr_id
    }

    /// Beam search on `layer` with width `ef`, returning candidates sorted ascending by distance.
    pub(crate) fn search_layer(
        &self,
        query: &[f32],
        entry_id: u64,
        ef: usize,
        layer: usize,
    ) -> Vec<HeapCandidate> {
        let Some(entry_node) = self.nodes.get(&entry_id) else {
            return Vec::new();
        };
        let entry_dist = self.metric.distance(query, &entry_node.vector);

        let mut visited = HashSet::new();
        visited.insert(entry_id);

        let mut candidates: BinaryHeap<Reverse<HeapCandidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<HeapCandidate> = BinaryHeap::new();

        let init = HeapCandidate {
            id: entry_id,
            distance: entry_dist,
        };
        candidates.push(Reverse(init.clone()));
        results.push(init);

        while let Some(Reverse(curr)) = candidates.pop() {
            let worst_dist = results.peek().map(|w| w.distance).unwrap_or(f32::INFINITY);
            if curr.distance > worst_dist && results.len() >= ef {
                break;
            }

            if let Some(node) = self.nodes.get(&curr.id) {
                if let Some(neighbors) = node.links.get(layer) {
                    for &nbr_id in neighbors {
                        if visited.insert(nbr_id) {
                            if let Some(nbr_node) = self.nodes.get(&nbr_id) {
                                let d = self.metric.distance(query, &nbr_node.vector);
                                let worst =
                                    results.peek().map(|w| w.distance).unwrap_or(f32::INFINITY);
                                if results.len() < ef || d < worst {
                                    let cand = HeapCandidate {
                                        id: nbr_id,
                                        distance: d,
                                    };
                                    candidates.push(Reverse(cand.clone()));
                                    results.push(cand);
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<HeapCandidate> = results.into_iter().collect();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id)));
        out
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_filter(query, k, None)
    }

    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let Some(mut ep) = self.entry_point else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        for lc in (1..=self.max_layer).rev() {
            ep = self.search_layer_greedy(query, ep, lc);
        }

        let ef = self.params.ef_search.max(k * 2);
        let candidates = self.search_layer(query, ep, ef, 0);

        let mut results = Vec::new();
        for c in candidates {
            if let Some(node) = self.nodes.get(&c.id) {
                if filter.is_none_or(|f| f.matches(&node.metadata)) {
                    results.push(SearchResult {
                        id: node.id,
                        distance: c.distance,
                        metadata: node.metadata.clone(),
                    });
                    if results.len() == k {
                        break;
                    }
                }
            }
        }
        Ok(results)
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
        assert!(counts[0] > 1700);
        assert!(counts[1] > 0);
        assert!(counts[0] > counts[1]);
    }

    #[test]
    fn test_hnsw_beam_search_finds_exact_nearest_neighbor() {
        let mut idx = HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap();
        for i in 0..100u64 {
            let v = vec![i as f32, (i * 2) as f32, 0.0, 1.0];
            idx.upsert(i, v, Metadata::new()).unwrap();
        }
        let hits = idx.search(&[42.1, 84.1, 0.0, 1.0], 3).unwrap();
        assert_eq!(hits[0].id, 42);
    }
}
