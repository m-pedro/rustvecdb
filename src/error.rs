use std::fmt;
use std::io;

#[derive(Debug)]
pub enum VecDbError {
    DimensionMismatch { expected: usize, actual: usize },
    EmptyVector,
    InvalidParameter(String),
    NotFound(u64),
    CorruptWal(String),
    Io(io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for VecDbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VecDbError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, actual)
            }
            VecDbError::EmptyVector => write!(f, "vector cannot have zero dimensions"),
            VecDbError::InvalidParameter(msg) => write!(f, "invalid parameter: {}", msg),
            VecDbError::NotFound(id) => write!(f, "vector id {} not found", id),
            VecDbError::CorruptWal(msg) => write!(f, "corrupt WAL: {}", msg),
            VecDbError::Io(e) => write!(f, "I/O error: {}", e),
            VecDbError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for VecDbError {}

impl From<io::Error> for VecDbError {
    fn from(e: io::Error) -> Self {
        VecDbError::Io(e)
    }
}

impl From<serde_json::Error> for VecDbError {
    fn from(e: serde_json::Error) -> Self {
        VecDbError::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, VecDbError>;
