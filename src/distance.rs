use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    L2,
    Cosine,
    DotProduct,
}

impl Metric {
    /// Computes a distance score where LOWER values always mean MORE SIMILAR vectors.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::DotProduct => -dot_product(a, b),
        }
    }
}

/// 4-way loop-unrolled squared L2 distance for LLVM NEON/AVX auto-vectorization.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);

    for (ca, cb) in (&mut chunks_a).zip(&mut chunks_b) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for (&xa, &xb) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        let d = xa - xb;
        total += d * d;
    }
    total
}

/// 4-way loop-unrolled dot product.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);

    for (ca, cb) in (&mut chunks_a).zip(&mut chunks_b) {
        sum0 += ca[0] * cb[0];
        sum1 += ca[1] * cb[1];
        sum2 += ca[2] * cb[2];
        sum3 += ca[3] * cb[3];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for (&xa, &xb) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        total += xa * xb;
    }
    total
}

/// Cosine distance in [0.0, 2.0]: `1.0 - cosine_similarity(a, b)`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();
    let denom = norm_a * norm_b;
    if denom <= f32::EPSILON {
        return 1.0;
    }
    (1.0 - (dot / denom)).clamp(0.0, 2.0)
}

/// Normalizes a vector in-place to unit L2 norm.
pub fn normalize_l2(v: &mut [f32]) {
    let norm = dot_product(v, v).sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_cosine_and_dot_kernels() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let orthogonal = vec![1.0, 0.0, 0.0, 0.0];
        let orth_b = vec![0.0, 1.0, 0.0, 0.0];

        assert!((Metric::L2.distance(&a, &b)).abs() < 1e-6);
        assert!((Metric::Cosine.distance(&a, &b)).abs() < 1e-6);
        assert!((Metric::DotProduct.distance(&a, &b) - (-55.0)).abs() < 1e-5);
        assert!((Metric::Cosine.distance(&orthogonal, &orth_b) - 1.0).abs() < 1e-6);
    }
}
