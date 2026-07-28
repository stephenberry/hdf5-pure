//! Streaming a MAT v7.3 file: `MatBuilder::finish_to` and the producer-backed
//! `write_blocks`.
//!
//! Two properties carry this feature, and every test here is one of them:
//!
//! - **Byte-identity.** A streamed file equals the buffered one, and a
//!   producer-backed dataset equals the materialized one. Without that, choosing
//!   the low-memory API silently changes the file and no known-good fixture
//!   applies to it.
//! - **Nothing is fully resident.** The producer is called during emission, once
//!   per block, into a buffer it shares with every other call. A refactor that
//!   reintroduced buffering would leave every correctness test green.

use hdf5_pure::mat::{Blocking, Compression, DataProducer, MatBuilder, MatError, Options};
use hdf5_pure::{AttrValue, File};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

/// What a [`DataProducer`] did, observed from the outside.
#[derive(Default, Debug)]
struct Log {
    /// Block indices, in the order they were asked for.
    calls: Vec<usize>,
    /// Address of the buffer handed over on each call. The writer promises one
    /// buffer for the whole dataset, and a reallocation would change this.
    buffers: Vec<usize>,
    /// Largest buffer length seen, i.e. the most data resident at once.
    high_water: usize,
}

/// Generates `f64` elements as a function of their linear index, so a dataset of
/// any size costs nothing to produce and the expected bytes are computable
/// without holding them.
struct Ramp {
    blocking: Blocking,
    log: Arc<Mutex<Log>>,
    /// Deliberately break the contract on this block, to prove the emitter
    /// checks rather than trusts.
    corrupt: Option<(usize, i64)>,
    /// Fail outright on this block.
    fail_at: Option<usize>,
}

impl Ramp {
    fn new(blocking: Blocking) -> (Self, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        (
            Self {
                blocking,
                log: Arc::clone(&log),
                corrupt: None,
                fail_at: None,
            },
            log,
        )
    }

    /// The value this producer yields for linear element `i`.
    fn value(i: u64) -> f64 {
        i as f64 * 0.5
    }
}

impl DataProducer for Ramp {
    fn block_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), MatError> {
        if self.fail_at == Some(index) {
            return Err(MatError::Custom("the source went away".into()));
        }
        let first = index as u64 * self.blocking.block_elements;
        let count = self.blocking.block_len(index) as u64 / 8;
        for i in 0..count {
            out.extend_from_slice(&Self::value(first + i).to_le_bytes());
        }
        match self.corrupt {
            Some((at, delta)) if at == index => {
                if delta < 0 {
                    out.truncate(out.len() - (-delta) as usize);
                } else {
                    out.extend_from_slice(&vec![0u8; delta as usize]);
                }
            }
            _ => {}
        }
        let mut log = self.log.lock().unwrap();
        log.calls.push(index);
        log.buffers.push(out.as_ptr() as usize);
        log.high_water = log.high_water.max(out.len());
        Ok(())
    }
}

/// Every element the ramp stands for, materialized. Only used to build the
/// buffered file a streamed one is compared against.
fn ramp_elements(n: u64) -> Vec<f64> {
    (0..n).map(Ramp::value).collect()
}

/// A `Write` that keeps the bytes' length but not the bytes, so a test can write
/// a dataset far larger than it would want resident.
#[derive(Default)]
struct Discard {
    written: u64,
}

impl std::io::Write for Discard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn read_f64(path: &std::path::Path, name: &str) -> Vec<f64> {
    let file = File::open(path).unwrap();
    file.dataset(name).unwrap().read_f64().unwrap()
}

fn read_class(path: &std::path::Path, name: &str) -> String {
    let file = File::open(path).unwrap();
    let attrs = file.dataset(name).unwrap().attrs().unwrap();
    match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        other => panic!("unexpected class attribute: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// finish_to
// ---------------------------------------------------------------------------

/// The whole point of the sink-based finish: it must not be a second way to
/// build a file, only a second way to deliver the same one. Content here covers
/// every part of the builder that used to run after `FileBuilder::finish` —
/// the MCOS subsystem, the `#refs#` group, and the userblock.
#[test]
fn a_streamed_file_is_byte_identical_to_a_buffered_one() {
    let build = || {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("plain", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        mb.write_char("text", "hello").unwrap();
        mb.write_string_object("modern", &["a".into(), "b".into()], &[2, 1])
            .unwrap();
        mb.struct_("nested", |s| {
            s.write_scalar_i32("n", 7)?;
            s.cell("items", &[2, 1], |c| {
                c.push_scalar_f64(1.5)?;
                c.push_string("two")?;
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
        mb
    };

    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();

    assert_eq!(
        buffered, streamed,
        "the streamed file must be byte-for-byte the buffered one"
    );
    assert_eq!(&buffered[..6], b"MATLAB", "userblock leads the file");
    assert_eq!(&buffered[124..128], &[0x00, 0x02, b'I', b'M']);
    assert_eq!(&buffered[512..516], b"\x89HDF", "superblock follows it");
}

/// The serde entry points are a second assembly path — `to_bytes` builds a
/// `FileBuilder` directly while `to_bytes_with_options` goes through
/// `MatBuilder` — so each needs its own streaming check.
#[test]
fn both_serde_paths_stream_the_bytes_they_return() {
    #[derive(serde::Serialize)]
    struct Payload {
        values: Vec<f64>,
        label: String,
    }
    let value = Payload {
        values: vec![1.0, 2.0, 3.0],
        label: "measurement".into(),
    };

    let buffered = hdf5_pure::mat::to_bytes(&value).unwrap();
    let mut streamed = Vec::new();
    hdf5_pure::mat::to_writer(&value, &mut streamed).unwrap();
    assert_eq!(buffered, streamed, "to_bytes / to_writer must agree");

    let options = Options::default();
    let buffered = hdf5_pure::mat::to_bytes_with_options(&value, &options).unwrap();
    let mut streamed = Vec::new();
    hdf5_pure::mat::to_writer_with_options(&value, &options, &mut streamed).unwrap();
    assert_eq!(
        buffered, streamed,
        "to_bytes_with_options / to_writer_with_options must agree"
    );
}

/// The three finalizers share one path, so a file delivered by any of them is
/// the same file. `write` is the one that hits the filesystem, which is where a
/// stray difference would show up first.
#[test]
fn every_finalizer_delivers_the_same_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("same.mat");

    let build = || {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("v", &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        mb.write_char("t", "abc").unwrap();
        mb
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    build().write(&path).unwrap();

    assert_eq!(buffered, streamed);
    assert_eq!(buffered, std::fs::read(&path).unwrap());
}

// ---------------------------------------------------------------------------
// write_blocks
// ---------------------------------------------------------------------------

/// If the two paths disagreed, the streamed one could not be reviewed against
/// any fixture built the ordinary way. Sized to span several blocks so the
/// blocking itself is exercised rather than a single-chunk special case.
#[test]
fn a_produced_dataset_is_byte_identical_to_a_materialized_one() {
    // MATLAB [4, 200_000]: a slice is 4 f64, so this spans several 1 MiB blocks.
    const DIMS: [usize; 2] = [4, 200_000];
    let n = (DIMS[0] * DIMS[1]) as u64;

    let mut mb = MatBuilder::new(Options::default());
    mb.write_f64("samples", &DIMS, &ramp_elements(n)).unwrap();
    let materialized = mb.finish().unwrap();

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, log) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(Options::default());
    let reported = mb
        .write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();
    let produced = mb.finish().unwrap();

    assert_eq!(reported, blocking, "plan must predict what staging chooses");
    assert!(
        blocking.block_count > 1,
        "the fixture must span more than one block, or it proves nothing about blocking"
    );
    assert_eq!(
        materialized, produced,
        "a produced dataset must be byte-for-byte a materialized one"
    );
    assert_eq!(log.lock().unwrap().calls.len(), blocking.block_count);
}

/// The memory property, asserted rather than inspected: the producer is called
/// only during emission, once per block in order, and always into the same
/// buffer — so at no point is more than one block resident.
#[test]
fn blocks_are_pulled_once_each_in_order_during_the_write() {
    const DIMS: [usize; 2] = [8, 500_000]; // 32 MB of f64
    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, log) = Ramp::new(blocking);

    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();
    assert!(
        log.lock().unwrap().calls.is_empty(),
        "staging must not touch the producer: layout works from the shape alone"
    );

    let mut sink = Discard::default();
    mb.finish_to(&mut sink).unwrap();

    let log = log.lock().unwrap();
    assert_eq!(
        log.calls,
        (0..blocking.block_count).collect::<Vec<_>>(),
        "every block exactly once, ascending"
    );
    assert!(
        sink.written > 32_000_000,
        "the whole dataset really was written ({} bytes)",
        sink.written
    );
    // One buffer for the whole dataset. Distinct addresses would mean a fresh
    // allocation per block, which is what the `&mut Vec<u8>` contract exists to
    // avoid; growing high water would mean blocks accumulating.
    let first = log.buffers[0];
    assert!(
        log.buffers.iter().all(|&b| b == first),
        "the producer must be handed the same buffer every time"
    );
    assert_eq!(
        log.high_water,
        blocking.block_len(0),
        "never more than one block resident"
    );
}

/// The blocking is chosen so that a block is a contiguous run of MATLAB's linear
/// element order. Reading back proves it: the elements must come out in the same
/// order a materialized write would put them.
#[test]
fn produced_elements_read_back_in_order() {
    const DIMS: [usize; 2] = [3, 7];
    let dir = tempdir().unwrap();
    let path = dir.path().join("ramp.mat");

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, _) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("ramp", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(read_f64(&path, "ramp"), ramp_elements(21));
    assert_eq!(read_class(&path, "ramp"), "double");
}

/// An element count that does not divide by the block size leaves a short last
/// block. It is written short, not padded — the data region is exactly the
/// dataset — so the dataset must read back at its declared length with nothing
/// appended.
#[test]
fn a_short_last_block_is_written_short() {
    // 3 x 100_001 = 300_003 f64 against a 131_072-element block: three blocks,
    // the last a partial remainder.
    const DIMS: [usize; 2] = [3, 100_001];
    let n = (DIMS[0] * DIMS[1]) as u64;
    let dir = tempdir().unwrap();
    let path = dir.path().join("uneven.mat");

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    assert!(
        blocking.last_block_elements < blocking.block_elements,
        "the fixture must actually leave a short tail, or it proves nothing"
    );
    let (producer, log) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("uneven", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    let read = read_f64(&path, "uneven");
    assert_eq!(read.len() as u64, n, "no padding leaked into the dataset");
    assert_eq!(read, ramp_elements(n));
    assert_eq!(
        log.lock().unwrap().calls,
        (0..blocking.block_count).collect::<Vec<_>>()
    );
}

/// A produced dataset is the same in a struct as at the root: `write_blocks`
/// resolves its target the way every other writer does.
#[test]
fn a_produced_dataset_can_live_inside_a_struct() {
    const DIMS: [usize; 2] = [2, 3];
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.mat");

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, _) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(Options::default());
    mb.struct_("outer", |s| {
        s.write_scalar_i32("tag", 1)?;
        Ok(())
    })
    .unwrap();
    mb.write_blocks::<f64>("top", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(read_f64(&path, "top"), ramp_elements(6));
    let file = File::open(&path).unwrap();
    assert!(file.dataset("outer/tag").is_ok());
}

/// A block of the wrong size shifts every address after it, so it is refused
/// instead of written. Both directions, because a long block and a short one
/// fail differently inside the emitter.
#[test]
fn a_block_of_the_wrong_size_is_refused() {
    const DIMS: [usize; 2] = [4, 100_000];
    for delta in [-8i64, 8] {
        let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
        let (mut producer, _) = Ramp::new(blocking);
        producer.corrupt = Some((1, delta));
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
            .unwrap();

        match mb.finish_to(Discard::default()) {
            Err(MatError::BlockSizeMismatch {
                block,
                expected,
                actual,
            }) => {
                assert_eq!(block, 1);
                assert_eq!(expected, blocking.block_len(1));
                assert_eq!(actual as i64, expected as i64 + delta);
            }
            other => panic!("expected BlockSizeMismatch for delta {delta}, got {other:?}"),
        }
    }
}

/// A producer's own failure has to survive the trip through the writer's
/// provider seam, which speaks a different error type. Flattening it to a
/// message would make a mid-write failure untraceable to its cause.
#[test]
fn a_producers_own_error_is_surfaced_verbatim() {
    const DIMS: [usize; 2] = [4, 100_000];
    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (mut producer, _) = Ramp::new(blocking);
    producer.fail_at = Some(1);

    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();

    match mb.finish_to(Discard::default()) {
        Err(MatError::Custom(msg)) => assert_eq!(msg, "the source went away"),
        other => panic!("expected the producer's own error, got {other:?}"),
    }
}

/// Compression is refused up front rather than silently dropped: a caller who
/// asked for deflate and got an unfiltered dataset would have no way to notice.
#[test]
fn compression_is_refused_rather_than_ignored() {
    let mut options = Options::default();
    options.compression = Compression::Deflate {
        level: 6,
        shuffle: false,
    };
    let blocking = Blocking::plan::<f64>(&[2, 4]).unwrap();
    let (producer, _) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(options);
    match mb.write_blocks::<f64>("x", &[2, 4], Box::new(producer)) {
        Err(MatError::CompressionUnsupportedForBlocks) => {}
        other => panic!("expected CompressionUnsupportedForBlocks, got {other:?}"),
    }
}

/// An empty shape writes the same marker every other writer does, and never
/// asks the producer for anything.
#[test]
fn an_empty_shape_writes_a_marker_and_asks_for_nothing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.mat");

    let materialized = {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("nothing", &[0, 0], &[]).unwrap();
        mb.finish().unwrap()
    };

    let blocking = Blocking::plan::<f64>(&[0, 0]).unwrap();
    let (producer, log) = Ramp::new(blocking);
    let mut mb = MatBuilder::new(Options::default());
    let reported = mb
        .write_blocks::<f64>("nothing", &[0, 0], Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(reported.block_count, 0);
    assert!(log.lock().unwrap().calls.is_empty());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        materialized,
        "an empty produced dataset must match an empty materialized one"
    );
}

/// The element type carries the MATLAB class, so each family has to name the
/// right one — a complex pair reports its *component* class, not a compound.
#[test]
fn element_types_name_their_matlab_class() {
    /// Yields a fixed byte pattern, whatever the element type.
    struct Bytes(Vec<u8>);
    impl DataProducer for Bytes {
        fn block_bytes(&self, _index: usize, out: &mut Vec<u8>) -> Result<(), MatError> {
            out.extend_from_slice(&self.0);
            Ok(())
        }
    }

    let dir = tempdir().unwrap();
    let path = dir.path().join("classes.mat");
    let mut mb = MatBuilder::new(Options::default());

    // 4 complex i16 = 4 * 4 bytes.
    mb.write_blocks::<(i16, i16)>("iq", &[2, 2], Box::new(Bytes(vec![0u8; 16])))
        .unwrap();
    mb.write_blocks::<bool>("flags", &[2, 2], Box::new(Bytes(vec![1u8; 4])))
        .unwrap();
    mb.write_blocks::<u32>("counts", &[2, 2], Box::new(Bytes(vec![0u8; 16])))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(read_class(&path, "iq"), "int16");
    assert_eq!(read_class(&path, "flags"), "logical");
    assert_eq!(read_class(&path, "counts"), "uint32");

    let file = File::open(&path).unwrap();
    let attrs = file.dataset("flags").unwrap().attrs().unwrap();
    assert!(
        attrs.contains_key("MATLAB_int_decode"),
        "a logical array carries its decode flag"
    );
}

/// A produced dataset must be the same file whichever way it is delivered, so
/// the two features compose rather than interacting.
#[test]
fn producing_and_streaming_compose() {
    const DIMS: [usize; 2] = [4, 50_000];
    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();

    let buffered = {
        let (producer, _) = Ramp::new(blocking);
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("s", &DIMS, Box::new(producer))
            .unwrap();
        mb.finish().unwrap()
    };
    let mut streamed = Vec::new();
    {
        let (producer, _) = Ramp::new(blocking);
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("s", &DIMS, Box::new(producer))
            .unwrap();
        mb.finish_to(&mut streamed).unwrap();
    }
    assert_eq!(buffered, streamed);
}
