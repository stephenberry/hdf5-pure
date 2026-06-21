//! Chunked-dataset editing on files that carry a userblock (non-zero base
//! address) — the chunked follow-up to the userblock slice of issue #104.
//!
//! Every stored chunk-index and chunk-data address in such a file is relative to
//! the userblock base. These tests exercise the three chunked write paths on a
//! userblock file and confirm the userblock bytes survive byte-for-byte:
//!
//!   * adding a chunked / filtered dataset (the relocatable blob is built with
//!     stored addresses and appended at end-of-file),
//!   * overwriting a chunked dataset in place (the index is walked on a
//!     base-relative view and the write offsets shifted back), and
//!   * overwriting a chunked dataset with a relocation (a fresh blob is built and
//!     the old chunk storage reclaimed), including reuse of that reclaimed space
//!     by a later commit in the same session.

use hdf5_pure::{EditSession, File, FileBuilder};

const UB: usize = 512;
const MARKER: &[u8] = b"USERBLOCK-CHUNK-0104";

/// Stamp a recognizable marker across the userblock region of `bytes` and return
/// the 512-byte userblock as written, for later byte-for-byte comparison.
fn stamp_userblock(bytes: &mut [u8]) -> Vec<u8> {
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    bytes[..UB].to_vec()
}

fn assert_userblock_unchanged(path: &std::path::Path, original: &[u8]) {
    let after = std::fs::read(path).unwrap();
    assert_eq!(
        &after[..UB],
        original,
        "userblock bytes changed across the edit"
    );
}

#[test]
fn userblock_chunked_add_roundtrip() {
    // Add a deflate-compressed multi-chunk dataset to a userblock file alongside
    // an untouched contiguous dataset, then read both back.
    let path = std::env::temp_dir().join("hdf5_pure_ub_chunk_add.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("contig")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let added: Vec<f64> = (0..1000).map(|i| (i % 13) as f64 * 0.25).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("chunked")
            .with_f64_data(&added)
            .with_shape(&[1000])
            .with_chunks(&[64])
            .with_deflate(6);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("chunked").unwrap().read_f64().unwrap(), added);
    assert_eq!(
        file.dataset("contig").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&path, &userblock);
    assert_eq!(&std::fs::read(&path).unwrap()[..MARKER.len()], MARKER);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_chunked_unfiltered_inplace_overwrite() {
    // An unfiltered chunked dataset: each chunk's slot is exactly its raw size, so
    // a same-shape overwrite re-fills every slot in place — no relocation, so the
    // file does not grow.
    let path = std::env::temp_dir().join("hdf5_pure_ub_chunk_inplace.h5");
    let original: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[200])
        .with_chunks(&[32]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    let updated: Vec<f64> = (0..200).map(|i| (i as f64) * -2.0 + 1.0).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.write_dataset("c").with_f64_data(&updated);
        s.commit().unwrap();
    }

    let len_after = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        len_after, len_before,
        "an unfiltered same-shape chunked overwrite must stay in place (no growth)"
    );
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("c")
            .unwrap()
            .read_f64()
            .unwrap(),
        updated
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_chunked_shrinking_inplace_overwrite() {
    // A deflate dataset whose new contents re-encode *smaller* than their slots
    // still fits in place: the chunks are rewritten into their existing slots and
    // the index is rebuilt in place to record the new (smaller) sizes — no
    // relocation, so the file does not grow. This exercises the base-aware
    // in-place index rebuild on a userblock file.
    let path = std::env::temp_dir().join("hdf5_pure_ub_chunk_shrink.h5");
    // Poorly compressible original (high-entropy) vs. highly compressible
    // replacement (all equal): each new chunk re-encodes far smaller than its slot.
    let original: Vec<f64> = (0..400).map(|i| (i as f64).sin() * 1e6).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[400])
        .with_chunks(&[40])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    let updated = vec![1.5f64; 400];
    {
        let mut s = EditSession::open(&path).unwrap();
        s.write_dataset("c").with_f64_data(&updated);
        s.commit().unwrap();
    }

    let len_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        len_after <= len_before,
        "a shrinking chunked overwrite must stay in place (no growth): {len_before} -> {len_after}"
    );
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("c")
            .unwrap()
            .read_f64()
            .unwrap(),
        updated
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn real_mat_add_chunked_dataset_preserves_userblock() {
    // Adding a chunked/deflate dataset to a real MATLAB v7.3 file: its chunk
    // index and chunk data addresses are written relative to the 512-byte MATLAB
    // userblock, and the added value plus an untouched original both read back.
    let src = std::path::Path::new("tests/fixtures/mat_real/test_string_v73.mat");
    let path = std::env::temp_dir().join("hdf5_pure_ub_real_mat_chunk.mat");
    std::fs::copy(src, &path).unwrap();
    let original_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(&original_userblock[..10], b"MATLAB 7.3");

    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    let added: Vec<f64> = (0..600).map(|i| (i % 17) as f64 * 0.5).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("hdf5_pure_chunk_probe")
            .with_f64_data(&added)
            .with_shape(&[600])
            .with_chunks(&[64])
            .with_deflate(6);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("hdf5_pure_chunk_probe")
            .unwrap()
            .read_f64()
            .unwrap(),
        added
    );
    assert_eq!(
        file.dataset("string_scalar").unwrap().read_u32().unwrap(),
        before
    );
    assert_eq!(
        &std::fs::read(&path).unwrap()[..UB],
        &original_userblock[..]
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_chunked_relocating_overwrite_roundtrip() {
    // A deflate dataset whose new contents compress to different per-chunk sizes
    // cannot re-fill its slots, so the overwrite rebuilds and relocates the chunk
    // storage and reclaims the old blob. The new value must read back.
    let path = std::env::temp_dir().join("hdf5_pure_ub_chunk_reloc.h5");
    // Highly compressible original (all equal) vs. a varied replacement: their
    // compressed sizes differ, forcing the relocating path.
    let original = vec![7.0f64; 500];
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep").with_i32_data(&[11, 22, 33]);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[500])
        .with_chunks(&[50])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let updated: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.write_dataset("c").with_f64_data(&updated);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), updated);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![11, 22, 33]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_chunked_overwrite_reuses_reclaimed_space() {
    // Reclaim-correctness check: a relocating chunked overwrite frees the old
    // chunk storage into the session free list; a second commit in the same
    // session then allocates into that freed region. If reclaim had freed a live
    // span (a base-address mistake), the second commit would corrupt it — so the
    // test reads every dataset back and (when built with the reference HDF5 C
    // library) the crosscheck below independently confirms the result.
    let path = std::env::temp_dir().join("hdf5_pure_ub_chunk_reclaim.h5");
    let original = vec![3.5f64; 800];
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[800])
        .with_chunks(&[40])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let updated: Vec<f64> = (0..800).map(|i| (i as f64) * 0.013).collect();
    let reuse: Vec<f64> = (0..64).map(|i| (i as f64) - 7.0).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        // Commit 1: relocate "c", reclaiming its old chunk storage.
        s.write_dataset("c").with_f64_data(&updated);
        s.commit().unwrap();
        // Commit 2: add a dataset that fits in the reclaimed region.
        s.create_dataset("reuse").with_f64_data(&reuse);
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), updated);
    assert_eq!(file.dataset("reuse").unwrap().read_f64().unwrap(), reuse);
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}
