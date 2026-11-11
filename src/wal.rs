use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecDbError};
use crate::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WalEntry {
    Upsert {
        id: u64,
        vector: Vec<f32>,
        metadata: Metadata,
    },
    Delete {
        id: u64,
    },
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, Vec<WalEntry>)> {
        let path = path.as_ref().to_path_buf();
        let mut entries = Vec::new();

        if path.exists() {
            let file = File::open(&path)?;
            let mut reader = BufReader::new(file);
            loop {
                let mut header = [0u8; 8];
                match reader.read_exact(&mut header) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(VecDbError::Io(e)),
                }

                let expected_crc = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

                let mut payload = vec![0u8; len];
                if let Err(e) = reader.read_exact(&mut payload) {
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        // Truncated tail frame from a crash; stop cleanly
                        break;
                    }
                    return Err(VecDbError::Io(e));
                }

                let actual_crc = crc32fast::hash(&payload);
                if actual_crc != expected_crc {
                    return Err(VecDbError::CorruptWal(format!(
                        "CRC mismatch: expected {:#010x}, got {:#010x}",
                        expected_crc, actual_crc
                    )));
                }

                let entry: WalEntry = serde_json::from_slice(&payload)?;
                entries.push(entry);
            }
        }

        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok((
            Self {
                path,
                writer: BufWriter::new(file),
            },
            entries,
        ))
    }

    pub fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let payload = serde_json::to_vec(entry)?;
        let crc = crc32fast::hash(&payload);
        let len = payload.len() as u32;

        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.writer.flush()?;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::AttrValue;
    use tempfile::tempdir;

    #[test]
    fn test_wal_append_replay_and_crc_verification() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("vectors.wal");

        {
            let (mut wal, initial) = Wal::open(&wal_path).unwrap();
            assert!(initial.is_empty());

            let mut meta = Metadata::new();
            meta.insert("doc".into(), AttrValue::Str("rust".into()));
            wal.append(&WalEntry::Upsert {
                id: 10,
                vector: vec![0.25, 0.75],
                metadata: meta,
            })
            .unwrap();
            wal.append(&WalEntry::Delete { id: 10 }).unwrap();
        }

        let (_, replayed) = Wal::open(&wal_path).unwrap();
        assert_eq!(replayed.len(), 2);
        assert!(matches!(replayed[0], WalEntry::Upsert { id: 10, .. }));
        assert_eq!(replayed[1], WalEntry::Delete { id: 10 });
    }
}
