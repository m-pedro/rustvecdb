use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Result, VecDbError};
use crate::metadata::{Filter, Metadata};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
    pub metadata: Metadata,
}

/// Internal heap entry ordered so the LARGEST distance sits at the top of the BinaryHeap,
/// allowing O(log k) eviction of the worst candidate when maintaining top-k closest vectors.
#[derive(Debug, Clone)]
pub(crate) struct HeapCandidate {
    pub id: u64,
    pub distance: f32,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance.to_bits() == other.distance.to_bits()
    }
}
impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    records: HashMap<u64, VectorRecord>,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim == 0 {
            return Err(VecDbError::EmptyVector);
        }
        Ok(Self {
            dim,
            metric,
            records: HashMap::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn upsert(&mut self, id: u64, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        self.records.insert(
            id,
            VectorRecord {
                id,
                vector,
                metadata,
            },
        );
        Ok(())
    }

    pub fn remove(&mut self, id: u64) -> bool {
        self.records.remove(&id).is_some()
    }

    pub fn get(&self, id: u64) -> Option<&VectorRecord> {
        self.records.get(&id)
    }

    pub fn search(
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
        if k == 0 || self.records.is_empty() {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

        for rec in self.records.values() {
            if let Some(f) = filter {
                if !f.matches(&rec.metadata) {
                    continue;
                }
            }
            let dist = self.metric.distance(query, &rec.vector);
            if heap.len() < k {
                heap.push(HeapEntry(HeapCandidate {
                    id: rec.id,
                    distance: dist,
                }));
            } else if let Some(worst) = heap.peek() {
                if dist < worst.0.distance {
                    heap.pop();
                    heap.push(HeapEntry(HeapCandidate {
                        id: rec.id,
                        distance: dist,
                    }));
                }
            }
        }

        let mut out: Vec<SearchResult> = heap
            .into_iter()
            .map(|entry| {
                let rec = &self.records[&entry.0.id];
                SearchResult {
                    id: rec.id,
                    distance: entry.0.distance,
                    metadata: rec.metadata.clone(),
                }
            })
            .collect();

        out.sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HeapEntry(HeapCandidate);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::AttrValue;

    #[test]
    fn test_flat_index_exact_knn_and_filter() {
        let mut idx = FlatIndex::new(3, Metric::L2).unwrap();
        let mut m1 = Metadata::new();
        m1.insert("tag".into(), AttrValue::Str("code".into()));
        let mut m2 = Metadata::new();
        m2.insert("tag".into(), AttrValue::Str("prose".into()));

        idx.upsert(1, vec![0.0, 0.0, 0.0], m1.clone()).unwrap();
        idx.upsert(2, vec![1.0, 1.0, 1.0], m2).unwrap();
        idx.upsert(3, vec![2.0, 2.0, 2.0], m1).unwrap();

        let res = idx.search(&[0.1, 0.1, 0.1], 2, None).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 1);
        assert_eq!(res[1].id, 2);

        // With filter tag == "code", id 2 is excluded so top-2 returns [1, 3]
        let f = Filter::Eq {
            field: "tag".into(),
            value: AttrValue::Str("code".into()),
        };
        let filtered = idx.search(&[0.1, 0.1, 0.1], 2, Some(&f)).unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id, 1);
        assert_eq!(filtered[1].id, 3);
    }
}
