//! Row-major to column-major transpose shared by both write paths.
//!
//! MATLAB stores 2-D arrays column-major; the serializer holds them row-major.
//! Both the default (`emit`) and options (`emit_with_builder`) write paths need
//! the same transpose, so it lives here once. The implementation is cache-tiled
//! (32x32 blocks) to keep both the strided source reads and the destination
//! writes cache-resident on large matrices.

/// Transpose a row-major matrix of shape `[rows, cols]` into column-major.
///
/// Element `(r, c)` at `row_major[r * cols + c]` lands at `out[c * rows + r]`.
fn transpose_2d<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    debug_assert_eq!(row_major.len(), rows * cols);
    let n = rows * cols;
    let mut out: Vec<T> = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }

    const BLK: usize = 32;
    let dst = out.as_mut_ptr();
    for cb in (0..cols).step_by(BLK) {
        let c_end = (cb + BLK).min(cols);
        for rb in (0..rows).step_by(BLK) {
            let r_end = (rb + BLK).min(rows);
            for r in rb..r_end {
                let src_row_base = r * cols;
                for c in cb..c_end {
                    let value = row_major[src_row_base + c];
                    // SAFETY: c < cols and r < rows so c*rows + r < cols*rows = n,
                    // and out has capacity n.
                    unsafe {
                        dst.add(c * rows + r).write(value);
                    }
                }
            }
        }
    }
    // SAFETY: every index 0..n was written above (each (r, c) maps to a unique
    // c * rows + r in 0..n).
    unsafe {
        out.set_len(n);
    }
    out
}

/// Transpose a row-major matrix of scalars into column-major order.
#[inline]
pub(super) fn transpose_scalars<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    transpose_2d(rows, cols, row_major)
}

/// Transpose a row-major matrix of `(re, im)` pairs into column-major order.
#[inline]
pub(super) fn transpose_pairs<T: Copy>(
    rows: usize,
    cols: usize,
    row_major: &[(T, T)],
) -> Vec<(T, T)> {
    transpose_2d(rows, cols, row_major)
}
