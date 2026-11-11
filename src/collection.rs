use std::path::Path;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::Result;
use crate::flat::{FlatIndex, SearchResult};
use crate::hnsw::{HnswIndex, HnswParams};
use crate::metadata::{Filter, Metadata};
use crate::wal::{Wal, WalEntry};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub vector_count: usize,
    pub dimension: usize,
    pub metric: Metric,
    pub hnsw_max_layer: usize,
}

struct CollectionInner {
    hnsw: HnswIndex,
    flat: FlatIndex,
    wal: Option<Wal>,
}

/// Thread-safe persistent vector collection backed by HNSW, FlatIndex, and a CRC32 WAL.
#[derive(Clone)]
pub struct Collection {
    inner: Arc<RwLock<CollectionInner>>,
}

impl Collection {
    pub fn new_in_memory(dim: usize, metric: Metric, params: HnswParams) -> Result<Self> {
        let hnsw = HnswIndex::new(dim, metric, params)?;
        let flat = FlatIndex::new(dim, metric)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(CollectionInner {
                hnsw,
                flat,
                wal: None,
            })),
        })
    }

    pub fn open_persistent(
        path: impl AsRef<Path>,
        dim: usize,
        metric: Metric,
        params: HnswParams,
    ) -> Result<Self> {
        let (wal, entries) = Wal::open(path)?;
        let mut hnsw = HnswIndex::new(dim, metric, params)?;
        let mut flat = FlatIndex::new(dim, metric)?;

        for entry in entries {
            match entry {
                WalEntry::Upsert {
                    id,
                    vector,
                    metadata,
                } => {
                    flat.upsert(id, vector.clone(), metadata.clone())?;
                    hnsw.upsert(id, vector, metadata)?;
                }
                WalEntry::Delete { id } => {
                    flat.remove(id);
                    hnsw.remove(id);
                }
            }
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(CollectionInner {
                hnsw,
                flat,
                wal: Some(wal),
            })),
        })
    }

    pub fn upsert(&self, id: u64, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let mut guard = self.inner.write().expect("collection write lock poisoned");
        if let Some(wal) = guard.wal.as_mut() {
            wal.append(&WalEntry::Upsert {
                id,
                vector: vector.clone(),
                metadata: metadata.clone(),
            })?;
        }
        guard.flat.upsert(id, vector.clone(), metadata.clone())?;
        guard.hnsw.upsert(id, vector, metadata)?;
        Ok(())
    }

    pub fn delete(&self, id: u64) -> Result<bool> {
        let mut guard = self.inner.write().expect("collection write lock poisoned");
        if let Some(wal) = guard.wal.as_mut() {
            wal.append(&WalEntry::Delete { id })?;
        }
        guard.flat.remove(id);
        Ok(guard.hnsw.remove(id))
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let guard = self.inner.read().expect("collection read lock poisoned");
        guard.hnsw.search_with_filter(query, k, filter)
    }

    pub fn search_exact(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let guard = self.inner.read().expect("collection read lock poisoned");
        guard.flat.search(query, k, filter)
    }

    pub fn stats(&self) -> CollectionStats {
        let guard = self.inner.read().expect("collection read lock poisoned");
        CollectionStats {
            vector_count: guard.hnsw.len(),
            dimension: guard.hnsw.dim(),
            metric: guard.hnsw.metric(),
            hnsw_max_layer: guard.hnsw.max_layer(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistent_collection_recovers_state_from_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("collection.wal");

        {
            let col = Collection::open_persistent(
                &wal_path,
                3,
                Metric::Cosine,
                HnswParams::default(),
            )
            .unwrap();
            col.upsert(1, vec![1.0, 0.0, 0.0], Metadata::new()).unwrap();
            col.upsert(2, vec![0.0, 1.0, 0.0], Metadata::new()).unwrap();
            col.upsert(3, vec![0.0, 0.0, 1.0], Metadata::new()).unwrap();
            col.delete(2).unwrap();
        }

        let reopened =
            Collection::open_persistent(&wal_path, 3, Metric::Cosine, HnswParams::default())
                .unwrap();
        assert_eq!(reopened.stats().vector_count, 2);
        let hits = reopened.search(&[0.9, 0.1, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].id, 1);
    }
}
