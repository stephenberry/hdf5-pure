//! Public chunk-cache observability: `Dataset::chunk_cache_stats()` lets a
//! downstream caller confirm their chunk-cache tuning is taking effect, without
//! reaching into crate internals.

use hdf5_pure::{ChunkCacheConfig, File, FileAccessOptions, FileBuilder};

fn chunked_file_bytes() -> Vec<u8> {
    let data: Vec<i32> = (0..256).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_i32_data(&data)
        .with_shape(&[256])
        .with_chunks(&[32]);
    b.finish().unwrap()
}

#[test]
fn fresh_handle_reports_empty_stats_before_any_read() {
    let file = File::from_bytes(chunked_file_bytes()).unwrap();
    let ds = file.dataset("chunked").unwrap();
    let stats = ds.chunk_cache_stats();
    assert!(!stats.index_loaded());
    assert_eq!(stats.cached_chunks(), 0);
    assert_eq!(stats.cached_bytes(), 0);
}

#[test]
fn enabled_cache_reports_retained_index_and_chunks_after_read() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessOptions::new().with_chunk_cache(ChunkCacheConfig::new()),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    let stats = ds.chunk_cache_stats();
    assert!(stats.index_loaded());
    assert!(stats.cached_chunks() > 0);
    assert_eq!(stats.cached_bytes(), stats.cached_chunks() * 32 * 4);
}

#[test]
fn disabled_cache_reports_nothing_retained_after_read() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled()),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    // Reads still return correct data; the cache simply retains nothing.
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    let stats = ds.chunk_cache_stats();
    assert!(!stats.index_loaded());
    assert_eq!(stats.cached_chunks(), 0);
    assert_eq!(stats.cached_bytes(), 0);
}
