//! Coalesced chunk reads for the streaming reader.
//!
//! # Why this exists
//!
//! The streaming chunked readers fetch one chunk per [`Source::read_exact_at`].
//! That is one `seek` plus one `read` per chunk, which is fine for the
//! megabyte-sized chunks a file written in one pass carries, and disastrous for
//! a file written as the data arrived: a recorder that appends a row at a time
//! produces one chunk per row, so a 2.7 MB file with 73 datasets over 280 rows
//! holds 20,440 chunks of 8 to 256 bytes. Reading all of it cost about four
//! times what the same read costs from a buffered [`crate::File::open`], and the
//! whole difference was read *granularity* — the file was read exactly once,
//! 90 bytes at a time.
//!
//! Those chunks are almost always adjacent on disk. libhdf5 allocates a chunk
//! when its chunk cache flushes, not when the write is issued, so even a writer
//! that fills 73 datasets a row at a time lays each dataset's chunks down in one
//! run (measured at 98.4% adjacent, and byte-identical to writing the datasets
//! one at a time). A reader that asks for the whole run in one call therefore
//! gets the same bytes in a few hundred reads instead of tens of thousands: on
//! the file above, 20,661 data reads became 616 and the read halved, from
//! 19.9 ms to 10.7 ms against 4.7 ms buffered.
//!
//! # What this does not do
//!
//! It does not turn a streaming read into a whole-file read. A span stops at
//! [`MAX_SPAN_BYTES`], so what this holds beyond the caller's own buffers is
//! bounded by that however large the dataset is — the reason `open_streaming`
//! exists is that a file larger than memory still opens, and this must not
//! spend that. The one span that may exceed the cap is a single chunk larger
//! than it, which is read alone, exactly as it was before. The cap is also what
//! keeps a windowed row read inside the peak `tests/windowed_read_memory.rs`
//! bounds.
//!
//! It also never reads a byte the caller did not ask for: only chunks that are
//! exactly adjacent are merged, so a coalesced read fetches the same bytes the
//! per-chunk reads would have. Bridging a gap between two nearby chunks would
//! trade read count for read volume; on the layouts this targets there is no gap
//! to bridge.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_read::ChunkInfo;
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::source::Source;

/// Largest span read in one call, and so the most this holds at a time.
///
/// The runs this coalesces are broken by the chunk index's own blocks every few
/// kilobytes, so on the motivating file 64 KiB already captured the whole win
/// (the mean span was 2.8 KB) and raising it changed nothing. The value is a
/// ceiling for the layouts where runs *are* long — a dataset of 4 KiB chunks
/// written in one pass — not a target.
pub(crate) const MAX_SPAN_BYTES: usize = 256 * 1024;

/// One contiguous byte range of the file covering one or more chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: u64,
    end: u64,
}

impl Span {
    fn covers(&self, start: u64, end: u64) -> bool {
        start >= self.start && end <= self.end
    }
}

/// Reads a dataset's chunks through coalesced spans.
///
/// Construct it with the chunks a single read will fetch — for a windowed read,
/// only the chunks the window overlaps, so a span never bridges chunks outside
/// it — then ask for each chunk's bytes with [`chunk_bytes`](Self::chunk_bytes).
/// A chunk in the same span as the last one is served from the buffer already
/// held; otherwise its whole span is read first.
pub(crate) struct ChunkSpanReader {
    /// Spans, sorted by `start` and non-overlapping.
    spans: Vec<Span>,
    /// The span currently held in `buf`.
    held: Option<Span>,
    buf: Vec<u8>,
    /// Holds a chunk that belongs to no span, so one such request does not
    /// evict the span the surrounding chunks are being served from.
    scratch: Vec<u8>,
}

impl ChunkSpanReader {
    /// Build a reader over `chunks`, or `None` when coalescing them would not
    /// pay.
    ///
    /// `None` means the caller should read each chunk directly, which is the
    /// answer whenever no two chunks are adjacent: every span would be a single
    /// chunk, and serving it through the buffer would only add a copy. Reading
    /// a chunk-per-row file whose writer had its chunk cache disabled — the one
    /// layout measured with no adjacency at all — pays nothing for this.
    pub(crate) fn new(chunks: &[ChunkInfo]) -> Option<Self> {
        if chunks.len() < 2 {
            return None;
        }

        let mut ranges: Vec<Span> = chunks
            .iter()
            .map(|c| Span {
                start: c.address,
                // A crafted address near `u64::MAX` saturates rather than
                // wrapping; the read itself is bounds-checked by the source.
                end: c.address.saturating_add(u64::from(c.chunk_size)),
            })
            .collect();
        ranges.sort_unstable_by_key(|s| s.start);

        let mut spans: Vec<Span> = Vec::new();
        for range in ranges {
            match spans.last_mut() {
                // Adjacent to (or contained in) the span being built, and still
                // within the budget measured from its start.
                Some(last)
                    if range.start <= last.end
                        && range.end.saturating_sub(last.start) <= MAX_SPAN_BYTES as u64 =>
                {
                    last.end = last.end.max(range.end);
                }
                _ => spans.push(range),
            }
        }

        // Nothing merged: every span is one chunk, so there is nothing to gain.
        if spans.len() == chunks.len() {
            return None;
        }
        Some(Self {
            spans,
            held: None,
            buf: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// The bytes of the chunk at `address`, reading its span if it is not the
    /// one already held.
    pub(crate) fn chunk_bytes<S: Source + ?Sized>(
        &mut self,
        source: &S,
        address: u64,
        len: usize,
    ) -> Result<&[u8], FormatError> {
        let end = address.saturating_add(len as u64);

        if !matches!(self.held, Some(span) if span.covers(address, end)) {
            let Some(span) = self.span_containing(address, end) else {
                // Not a chunk this reader was built over. Serve it directly and
                // leave the held span alone.
                self.scratch = source.read_exact_at(address, len)?;
                return Ok(&self.scratch);
            };
            self.buf = source.read_exact_at(span.start, (span.end - span.start).to_usize()?)?;
            self.held = Some(span);
        }

        let span = self.held.expect("held span set above");
        let lo = (address - span.start).to_usize()?;
        self.buf
            .get(lo..lo + len)
            .ok_or(FormatError::UnexpectedEof {
                expected: lo + len,
                available: self.buf.len(),
            })
    }

    /// The span covering `[start, end)`, if this reader was built over a chunk
    /// there.
    fn span_containing(&self, start: u64, end: u64) -> Option<Span> {
        let idx = match self.spans.binary_search_by_key(&start, |s| s.start) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        self.spans
            .get(idx)
            .copied()
            .filter(|s| s.covers(start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::BytesSource;
    use core::cell::Cell;

    /// A source that records every read it serves.
    struct CountingSource {
        bytes: Vec<u8>,
        reads: Cell<usize>,
        bytes_read: Cell<usize>,
        largest: Cell<usize>,
    }

    impl CountingSource {
        fn new(len: usize) -> Self {
            Self {
                bytes: (0..len).map(|i| i as u8).collect(),
                reads: Cell::new(0),
                bytes_read: Cell::new(0),
                largest: Cell::new(0),
            }
        }
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.set(self.reads.get() + 1);
            self.bytes_read.set(self.bytes_read.get() + buf.len());
            self.largest.set(self.largest.get().max(buf.len()));
            BytesSource::new(&self.bytes).read_at(offset, buf)
        }
    }

    /// `count` chunks of `size` bytes laid end to end from `start`.
    fn run(start: u64, size: u32, count: usize) -> Vec<ChunkInfo> {
        (0..count)
            .map(|i| ChunkInfo {
                chunk_size: size,
                filter_mask: 0,
                offsets: vec![i as u64],
                address: start + i as u64 * u64::from(size),
            })
            .collect()
    }

    #[test]
    fn one_read_serves_a_whole_run_of_chunks() {
        let chunks = run(16, 8, 100);
        let source = CountingSource::new(4096);
        let mut reader = ChunkSpanReader::new(&chunks).expect("run of adjacent chunks coalesces");

        for chunk in &chunks {
            let bytes = reader.chunk_bytes(&source, chunk.address, 8).unwrap();
            let expected: Vec<u8> = (chunk.address..chunk.address + 8)
                .map(|b| b as u8)
                .collect();
            assert_eq!(bytes, &expected[..], "chunk at {}", chunk.address);
        }

        // One read for the whole run, and not a byte beyond it.
        assert_eq!(source.reads.get(), 1);
        assert_eq!(source.bytes_read.get(), 800);
    }

    #[test]
    fn a_long_run_is_split_at_the_budget() {
        // 40 chunks of 8 KiB is 320 KiB: more than one span, none over budget.
        let chunks = run(0, 8192, 40);
        let source = CountingSource::new(40 * 8192);
        let mut reader = ChunkSpanReader::new(&chunks).expect("adjacent chunks coalesce");

        for chunk in &chunks {
            let bytes = reader
                .chunk_bytes(&source, chunk.address, chunk.chunk_size as usize)
                .unwrap();
            assert_eq!(bytes.len(), 8192);
            assert_eq!(bytes[0], chunk.address as u8);
        }

        assert!(source.reads.get() > 1, "the run must not be read in one go");
        assert!(
            source.largest.get() <= MAX_SPAN_BYTES,
            "a span read {} bytes, over the {MAX_SPAN_BYTES}-byte budget",
            source.largest.get()
        );
        // Still exactly the chunk bytes, spread over the spans.
        assert_eq!(source.bytes_read.get(), 40 * 8192);
    }

    #[test]
    fn chunks_that_are_not_adjacent_are_left_to_the_direct_path() {
        let scattered: Vec<ChunkInfo> = (0..16)
            .map(|i| ChunkInfo {
                chunk_size: 8,
                filter_mask: 0,
                offsets: vec![i],
                address: i * 4096, // a gap between every pair
            })
            .collect();
        assert!(ChunkSpanReader::new(&scattered).is_none());
    }

    #[test]
    fn a_single_chunk_is_left_to_the_direct_path() {
        assert!(ChunkSpanReader::new(&run(0, 64, 1)).is_none());
    }

    #[test]
    fn a_partly_adjacent_layout_still_coalesces_its_runs() {
        // Two runs of four, far apart: two spans, so two reads for eight chunks.
        let mut chunks = run(0, 32, 4);
        chunks.extend(run(100_000, 32, 4));
        let source = CountingSource::new(200_000);
        let mut reader = ChunkSpanReader::new(&chunks).expect("two runs coalesce");

        for chunk in &chunks {
            reader.chunk_bytes(&source, chunk.address, 32).unwrap();
        }
        assert_eq!(source.reads.get(), 2);
        assert_eq!(source.bytes_read.get(), 8 * 32);
    }

    #[test]
    fn a_chunk_outside_every_span_is_read_directly() {
        let chunks = run(0, 32, 8);
        let source = CountingSource::new(200_000);
        let mut reader = ChunkSpanReader::new(&chunks).expect("adjacent chunks coalesce");

        reader.chunk_bytes(&source, 0, 32).unwrap();
        assert_eq!(source.reads.get(), 1);

        // A chunk this reader was not built over still reads correctly...
        let stray = reader.chunk_bytes(&source, 100_000, 4).unwrap().to_vec();
        assert_eq!(stray, vec![160u8, 161, 162, 163]);
        assert_eq!(source.reads.get(), 2);

        // ...and does not cost the held span, which still serves its chunks.
        reader.chunk_bytes(&source, 224, 32).unwrap();
        assert_eq!(source.reads.get(), 2);
    }

    #[test]
    fn a_chunk_larger_than_the_budget_is_read_on_its_own() {
        // One 300 KiB chunk, then a run of small ones. The big chunk cannot
        // join a span, so it is read alone — the same read it would have been
        // without coalescing — and does not drag its neighbours over budget.
        let big = MAX_SPAN_BYTES + 40 * 1024;
        let mut chunks = vec![ChunkInfo {
            chunk_size: big as u32,
            filter_mask: 0,
            offsets: vec![0],
            address: 0,
        }];
        chunks.extend(run(big as u64, 64, 8));

        let source = CountingSource::new(big + 8 * 64);
        let mut reader = ChunkSpanReader::new(&chunks).expect("the small chunks coalesce");

        assert_eq!(reader.chunk_bytes(&source, 0, big).unwrap().len(), big);
        assert_eq!(source.largest.get(), big, "the big chunk was read alone");

        for chunk in &chunks[1..] {
            reader.chunk_bytes(&source, chunk.address, 64).unwrap();
        }
        // The big chunk, then one span over the eight small ones.
        assert_eq!(source.reads.get(), 2);
        assert_eq!(source.bytes_read.get(), big + 8 * 64);
    }

    #[test]
    fn chunks_are_served_whatever_order_they_are_asked_for() {
        // The readers walk chunks in index order, which is not always address
        // order; a span already held must serve either way.
        let chunks = run(64, 16, 8);
        let source = CountingSource::new(1024);
        let mut reader = ChunkSpanReader::new(&chunks).expect("adjacent chunks coalesce");

        for chunk in chunks.iter().rev() {
            let bytes = reader.chunk_bytes(&source, chunk.address, 16).unwrap();
            assert_eq!(bytes[0], chunk.address as u8);
        }
        assert_eq!(source.reads.get(), 1);
    }
}
