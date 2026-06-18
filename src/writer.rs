//! Writing API: FileBuilder and GroupBuilder for creating HDF5 files.

use std::io::Write;

use crate::chunked_write::ByteSink;
use crate::file_writer::FileWriter as FormatWriter;
use crate::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::error::{Error, FormatError};
use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;

/// Builder for creating a new HDF5 file.
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::FileBuilder;
/// use hdf5_pure::AttrValue;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
/// builder.set_attr("version", AttrValue::I64(1));
/// builder.write("output.h5").unwrap();
/// ```
pub struct FileBuilder {
    writer: FormatWriter,
}

impl FileBuilder {
    /// Create a new file builder.
    pub fn new() -> Self {
        Self {
            writer: FormatWriter::new(),
        }
    }

    /// Create a dataset at the root level. Returns a mutable reference to
    /// a `DatasetBuilder` for configuring data, shape, and attributes.
    pub fn create_dataset(&mut self, name: &str) -> &mut FormatDatasetBuilder {
        self.writer.create_dataset(name)
    }

    /// Create a group builder. Call `.finish()` on the returned builder
    /// to complete it, then pass to `add_group()`.
    pub fn create_group(&mut self, name: &str) -> FormatGroupBuilder {
        self.writer.create_group(name)
    }

    /// Add a finished group to the file.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.writer.add_group(group);
    }

    /// Set the userblock size in bytes. Must be a power of two >= 512 or 0 (no userblock).
    /// The userblock region is filled with zeros. After calling `finish()`, write your
    /// userblock data into `bytes[0..size]`.
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.writer.with_userblock(size);
        self
    }

    /// Constrain the on-disk format version of the file, mirroring HDF5's
    /// `H5Pset_libver_bounds`. The produced file must fall within `[low, high]`,
    /// or [`finish`](Self::finish) / [`write`](Self::write) fails with
    /// [`Error::Format`] wrapping
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    ///
    /// This crate writes exactly one format — the version 3 superblock from
    /// HDF5 1.10 ([`LibVer::WRITER_OUTPUT`]) — so this is a compatibility
    /// assertion, not a format selector: a bound that excludes 1.10 (an upper
    /// bound older than it, or a lower bound newer than it) is rejected.
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.writer.with_libver_bounds(low, high);
        self
    }

    /// Set the file-space management strategy, mirroring HDF5's
    /// `H5Pset_file_space_strategy`. The strategy, persist flag, and free-space
    /// section `threshold` are recorded in the file's superblock extension, so
    /// the reference C library and a later reopen observe the choice.
    ///
    /// `persist = true` records that freed space should be tracked on disk across
    /// closes. A brand-new file has nothing to track, so this only records the
    /// intent; freeing space in a later [`EditSession`](crate::EditSession) then
    /// writes the on-disk free-space-manager blocks that survive a reopen.
    pub fn with_file_space_strategy(
        &mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> &mut Self {
        self.writer
            .with_file_space_strategy(strategy, persist, threshold);
        self
    }

    /// Set the file-space page size, mirroring HDF5's
    /// `H5Pset_file_space_page_size`. Recorded in the superblock extension.
    pub fn with_file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.writer.with_file_space_page_size(page_size);
        self
    }

    /// Set an attribute on the root group.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.writer.set_root_attr(name, value);
    }

    /// Serialize the file to bytes in memory.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.writer.finish()?)
    }

    /// Serialize the file directly to a [`Write`] sink, without first buffering
    /// the whole file in memory.
    ///
    /// Produces byte-for-byte the same file as [`finish`](Self::finish), but a
    /// dataset staged for verbatim chunk *streaming* (repack's out-of-core path)
    /// has its chunks pulled from the source and written one at a time, so peak
    /// memory stays bounded by a single chunk plus the file metadata rather than
    /// the whole dataset. The sink is written front-to-back, so it need not be
    /// seekable.
    pub fn finish_to<W: Write>(self, w: W) -> Result<(), Error> {
        let mut sink = WriteSink::new(std::io::BufWriter::new(w));
        if let Err(fe) = self.writer.finish_to_sink(&mut sink) {
            // If the failure came from the sink's I/O, surface the real
            // `io::Error`; otherwise it is a genuine format error.
            return match sink.err.take() {
                Some(io_err) => Err(Error::Io(io_err)),
                None => Err(Error::Format(fe)),
            };
        }
        sink.into_inner().flush().map_err(Error::Io)
    }

    /// Serialize and write the file to the given path.
    ///
    /// Streams the file to disk (see [`finish_to`](Self::finish_to)), so a repack
    /// staging streamed chunks does not hold the whole output in memory.
    pub fn write<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), Error> {
        let file = std::fs::File::create(path).map_err(Error::Io)?;
        self.finish_to(file)
    }
}

/// Adapts a [`std::io::Write`] to the writer's [`ByteSink`] so a file can be
/// assembled straight onto the sink. Because `ByteSink` is `no_std` and cannot
/// carry a `std::io::Error`, an I/O failure is stashed here and the surrounding
/// [`FileBuilder::finish_to`] turns it back into [`Error::Io`].
struct WriteSink<W: Write> {
    inner: W,
    written: u64,
    err: Option<std::io::Error>,
}

impl<W: Write> WriteSink<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            written: 0,
            err: None,
        }
    }

    fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> ByteSink for WriteSink<W> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        match self.inner.write_all(bytes) {
            Ok(()) => {
                self.written += bytes.len() as u64;
                Ok(())
            }
            Err(e) => {
                self.err = Some(e);
                // A placeholder format error; `finish_to` replaces it with the
                // stashed `io::Error` above, so its message is never surfaced.
                Err(FormatError::SerializationError(
                    "streaming output write failed".into(),
                ))
            }
        }
    }

    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError> {
        // Emit padding in bounded blocks so a large userblock never allocates a
        // matching buffer.
        const ZEROS: [u8; 4096] = [0u8; 4096];
        let mut remaining = n;
        while remaining > 0 {
            let take = remaining.min(ZEROS.len());
            self.put(&ZEROS[..take])?;
            remaining -= take;
        }
        Ok(())
    }

    fn position(&self) -> u64 {
        self.written
    }
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::chunked_write::{ChunkMeta, ChunkProvider};
    use std::sync::{Arc, Mutex};

    type Calls = Arc<Mutex<Vec<usize>>>;

    /// A test [`ChunkProvider`] serving fixed in-memory chunk bytes, recording
    /// the order of `chunk_bytes` calls so a test can assert the streaming
    /// writer pulls each chunk exactly once, in ascending slot order. With
    /// `short` set, slot 0 returns one byte fewer than planned (size-mismatch).
    /// `Arc<Mutex<_>>` (not `Rc<RefCell<_>>`) keeps it `Send + Sync`, as the
    /// `ChunkProvider` supertrait requires.
    struct MemProvider {
        chunks: Vec<Vec<u8>>,
        calls: Calls,
        short: bool,
    }

    impl ChunkProvider for MemProvider {
        fn chunk_bytes(&self, index: usize) -> Result<Vec<u8>, FormatError> {
            self.calls.lock().unwrap().push(index);
            let mut bytes = self.chunks[index].clone();
            if self.short && index == 0 {
                bytes.pop();
            }
            Ok(bytes)
        }
    }

    fn f64_chunk(vals: &[f64]) -> Vec<u8> {
        let mut v = Vec::new();
        for &x in vals {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    /// Build a file with one lazily-streamed, unfiltered chunked f64 dataset.
    /// Unfiltered means the "compressed" bytes are the raw element bytes, so the
    /// produced file is a plain chunked f64 dataset that reads back.
    fn build_lazy(
        chunk_bytes: Vec<Vec<u8>>,
        dims: &[u64],
        chunk_dims: &[u64],
        maxshape: Option<&[u64]>,
        calls: Calls,
        short: bool,
    ) -> FileBuilder {
        let meta: Vec<ChunkMeta> = chunk_bytes
            .iter()
            .map(|c| ChunkMeta {
                compressed_size: c.len() as u64,
                filter_mask: 0,
            })
            .collect();
        let provider = MemProvider {
            chunks: chunk_bytes,
            calls,
            short,
        };
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_raw_chunks_lazy(
            crate::type_builders::make_f64_type(),
            dims,
            maxshape,
            chunk_dims,
            8,
            None,
            meta,
            Box::new(provider),
        );
        b
    }

    fn read_back_f64(bytes: Vec<u8>) -> Vec<f64> {
        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let raw = file.dataset("d").unwrap().read_raw().unwrap();
        raw.chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn streamed_output_matches_buffered_and_streams_one_chunk_at_a_time() {
        let chunks = vec![
            f64_chunk(&[1.0, 2.0]),
            f64_chunk(&[3.0, 4.0]),
            f64_chunk(&[5.0, 6.0]),
        ];

        let calls_buf = Arc::new(Mutex::new(Vec::new()));
        let buffered = build_lazy(chunks.clone(), &[6], &[2], None, calls_buf.clone(), false)
            .finish()
            .unwrap();

        let calls_str = Arc::new(Mutex::new(Vec::new()));
        let mut streamed = Vec::new();
        build_lazy(chunks.clone(), &[6], &[2], None, calls_str.clone(), false)
            .finish_to(&mut streamed)
            .unwrap();

        // The streaming (io::Write) path and the buffered (Vec) path must produce
        // byte-for-byte the same file.
        assert_eq!(
            buffered, streamed,
            "streamed output must be byte-identical to buffered output"
        );
        // Each chunk is pulled exactly once, in ascending slot order — i.e. the
        // writer streams chunk-by-chunk rather than collecting them all.
        assert_eq!(*calls_buf.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(*calls_str.lock().unwrap(), vec![0, 1, 2]);
        // And the file reads back to the original values.
        assert_eq!(read_back_f64(buffered), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn file_builder_keeps_its_auto_traits() {
        // The lazy chunk provider is boxed into `FileBuilder`; a bare boxed trait
        // object would strip `Send`/`Sync` (fixed by the `ChunkProvider`
        // supertrait) and `UnwindSafe`/`RefUnwindSafe` (fixed by wrapping it in
        // `AssertUnwindSafe`). Removing any of these auto-trait impls is a semver
        // break that `cargo-semver-checks` enforces in CI, so pin all four here.
        fn assert_auto_traits<
            T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe,
        >() {
        }
        assert_auto_traits::<FileBuilder>();
    }

    #[test]
    fn streaming_writer_rejects_provider_size_mismatch() {
        let chunks = vec![f64_chunk(&[1.0, 2.0]), f64_chunk(&[3.0, 4.0])];
        let calls = Arc::new(Mutex::new(Vec::new()));
        // `short` makes slot 0's provider return fewer bytes than the planned
        // size; the emitter must reject the desync rather than write a corrupt file.
        let err = build_lazy(chunks, &[4], &[2], None, calls, true)
            .finish()
            .unwrap_err();
        match err {
            Error::Format(FormatError::ChunkedReadError(_)) => {}
            other => panic!("expected ChunkedReadError, got {other:?}"),
        }
    }

    /// Assert the buffered and streamed outputs are byte-identical for one
    /// chunked layout, and that the produced file reads back.
    fn assert_variant_streams_identically(
        chunks: Vec<Vec<u8>>,
        dims: &[u64],
        chunk_dims: &[u64],
        maxshape: Option<&[u64]>,
    ) {
        let buffered = build_lazy(
            chunks.clone(),
            dims,
            chunk_dims,
            maxshape,
            Arc::new(Mutex::new(Vec::new())),
            false,
        )
        .finish()
        .unwrap();
        let mut streamed = Vec::new();
        build_lazy(
            chunks,
            dims,
            chunk_dims,
            maxshape,
            Arc::new(Mutex::new(Vec::new())),
            false,
        )
        .finish_to(&mut streamed)
        .unwrap();
        assert_eq!(
            buffered, streamed,
            "index variant dims={dims:?} chunk={chunk_dims:?} must stream identically"
        );
        // Sanity: the produced file reads back.
        let _ = read_back_f64(buffered);
    }

    #[test]
    fn streamed_equals_buffered_across_index_variants() {
        // single-chunk, fixed-array (>1 chunk), and extensible-array (unlimited
        // max shape) all lay out from sizes alone, so each must stream identically.
        assert_variant_streams_identically(vec![f64_chunk(&[1.0, 2.0])], &[2], &[2], None);
        assert_variant_streams_identically(
            (0..5)
                .map(|i| f64_chunk(&[i as f64, i as f64 + 0.5]))
                .collect(),
            &[10],
            &[2],
            None,
        );
        assert_variant_streams_identically(
            vec![f64_chunk(&[1.0, 2.0]), f64_chunk(&[3.0, 4.0])],
            &[4],
            &[2],
            Some(&[u64::MAX]),
        );
    }
}
