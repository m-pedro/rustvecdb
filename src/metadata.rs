use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

pub type Metadata = HashMap<String, AttrValue>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    Eq { field: String, value: AttrValue },
    Ne { field: String, value: AttrValue },
    Gt { field: String, value: f64 },
    Ge { field: String, value: f64 },
    Lt { field: String, value: f64 },
    Le { field: String, value: f64 },
    In { field: String, values: Vec<AttrValue> },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub fn matches(&self, meta: &Metadata) -> bool {
        match self {
            Filter::Eq { field, value } => meta.get(field) == Some(value),
            Filter::Ne { field, value } => meta.get(field).is_some_and(|v| v != value),
            Filter::Gt { field, value } => extract_num(meta.get(field)).is_some_and(|n| n > *value),
            Filter::Ge { field, value } => {
                extract_num(meta.get(field)).is_some_and(|n| n >= *value)
            }
            Filter::Lt { field, value } => extract_num(meta.get(field)).is_some_and(|n| n < *value),
            Filter::Le { field, value } => {
                extract_num(meta.get(field)).is_some_and(|n| n <= *value)
            }
            Filter::In { field, values } => {
                meta.get(field).is_some_and(|v| values.iter().any(|x| x == v))
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(meta)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(meta)),
            Filter::Not(inner) => !inner.matches(meta),
        }
    }
}

fn extract_num(val: Option<&AttrValue>) -> Option<f64> {
    match val? {
        AttrValue::Int(i) => Some(*i as f64),
        AttrValue::Float(f) => Some(*f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_compound_filter_evaluation() {
        let mut meta = Metadata::new();
        meta.insert("category".into(), AttrValue::Str("gpu".into()));
        meta.insert("vram_gb".into(), AttrValue::Int(24));
        meta.insert("in_stock".into(), AttrValue::Bool(true));

        let filter = Filter::And(vec![
            Filter::Eq {
                field: "category".into(),
                value: AttrValue::Str("gpu".into()),
            },
            Filter::Ge {
                field: "vram_gb".into(),
                value: 16.0,
            },
            Filter::Not(Box::new(Filter::Eq {
                field: "in_stock".into(),
                value: AttrValue::Bool(false),
            })),
        ]);

        assert!(filter.matches(&meta));

        meta.insert("vram_gb".into(), AttrValue::Int(8));
        assert!(!filter.matches(&meta));
    }
}
