//! Replay every prefix of what reached the disk, and read what it left.
//!
//! # The defect class this exists for
//!
//! Since the write gathering of issue #288, the writes an operation makes
//! between two ordering barriers are issued in **address order** rather than in
//! the order the engine made them. In this format that is systematically the
//! wrong way round: every publish point sits at a *lower* address than the
//! content it names — the superblock at address 0, an object header's dataspace
//! dimension and Extensible-Array element count near the front of the file, the
//! chunk bytes and index blocks they name at end-of-file. So an
//! append-then-publish pair with no barrier between them now publishes *first*.
//!
//! The resulting failure is the bad kind. A pointer that survives while the
//! block it names does not is a checksummed, valid-looking address into bytes
//! past end-of-file, and the *next* append believes the block exists and writes
//! into nothing.
//!
//! What holds this together is [`FileImage::ordering_barrier`], and a missing one
//! is invisible to every ordinary test: the operation completes, the file is
//! correct, and only a machine that stops at the wrong instant can tell. This
//! module is what stops the machine at every instant.
//!
//! # How it works
//!
//! [`disk_log`] records each write and each `set_len` that a session actually
//! issued. A [`Recording`] pairs that log with the file's bytes as they were
//! before the session opened, which makes every prefix of the log a file that
//! could really have existed: the operations are positioned, so applying the
//! first *k* of them to the starting bytes is exactly the disk state after *k*
//! of them completed.
//!
//! Each prefix is then read back and classified:
//!
//! - **clean** — it reads, and returns a state a crash at that point may leave.
//! - **loud** — it refuses to read. This is the *benign* outcome: the caller is
//!   told, rather than handed something wrong.
//! - **silent** — it reads without complaint and returns something else. This is
//!   the outcome the barriers exist to prevent, and any occurrence fails.
//!
//! # This one can fail
//!
//! A crash harness that cannot fail is decoration. Two things keep this one
//! honest. Deleting either ordering barrier in
//! [`crate::chunk_index_inplace`] makes
//! [`appending_survives_a_crash_at_every_write`] report the unreachable block
//! those barriers exist to prevent. And the driver asserts a floor on how many
//! prefixes came back **clean**, because a checker that called everything loud,
//! or a workload that stopped doing any work, would otherwise report no silent
//! corruption and pass.

use std::path::Path;

use crate::edit::{AppendBuilder, AppendTarget, MemoryStrategy, SyncPolicy, WriteEngine};
use crate::error::Error;
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpaceStrategy;
use crate::image::disk_log::{self, DiskOp};
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

/// What reading one replayed prefix made of it.
#[derive(Debug)]
enum Verdict {
    /// Read back, and the state is one a crash at this point may leave.
    Clean,
    /// Refused to read. The benign failure: the caller is told.
    Loud(String),
    /// Read back without complaint, and returned something else.
    Silent(String),
}

impl Verdict {
    /// Classify a read. An `Error` is [`Loud`](Verdict::Loud) whatever it says —
    /// the file declined to hand over data, which is the outcome that cannot
    /// mislead. Only a *successful* read is judged on its contents.
    fn of<T>(read: Result<T, Error>, judge: impl FnOnce(T) -> Result<(), String>) -> Verdict {
        match read {
            Err(e) => Verdict::Loud(std::format!("{e:?}")),
            Ok(v) => match judge(v) {
                Ok(()) => Verdict::Clean,
                Err(why) => Verdict::Silent(why),
            },
        }
    }
}

/// A session's disk operations, and the file they started from.
struct Recording {
    label: String,
    /// The file's bytes before the recorded session opened it. Read before the
    /// session exists rather than alongside it, so no platform's file locking
    /// has an opinion about it.
    base: Vec<u8>,
    ops: Vec<DiskOp>,
}

impl Recording {
    /// Record everything `work` puts on the disk. `work` is handed `path`, and
    /// must open and close its own session inside the call: a session's teardown
    /// writes are part of what a crash can interrupt, so they belong in the log.
    fn of(label: &str, path: &Path, work: impl FnOnce(&Path)) -> Self {
        let base = std::fs::read(path).expect("the starting file");
        disk_log::start();
        work(path);
        let ops = disk_log::take();
        assert!(
            !ops.is_empty(),
            "{label}: recorded nothing, so there is no crash point to replay"
        );
        Self {
            label: label.to_string(),
            base,
            ops,
        }
    }

    /// Apply one operation to an in-memory image of the file, exactly as the
    /// filesystem would: a positioned write past end-of-file extends it, leaving
    /// the gap reading as zeros, and `set_len` both truncates and extends.
    fn apply(image: &mut Vec<u8>, op: &DiskOp) {
        match op {
            DiskOp::Write { offset, bytes } => {
                let start = *offset as usize;
                let end = start + bytes.len();
                if image.len() < end {
                    image.resize(end, 0);
                }
                image[start..end].copy_from_slice(bytes);
            }
            DiskOp::SetLen(len) => image.resize(*len as usize, 0),
        }
    }

    /// Materialize every prefix in turn and hand each to `check`.
    ///
    /// `min_clean` is the floor this run must meet, as a fraction of the prefixes
    /// replayed. It is what makes a vacuous pass fail: a workload that stopped
    /// doing work, or a checker that called every state loud, would report no
    /// silent corruption and otherwise sail through.
    fn replay_every_prefix(
        &self,
        dir: &Path,
        min_clean: f64,
        check: impl Fn(&Path) -> Verdict,
    ) -> Tally {
        let path = dir.join(std::format!("{}.replay.h5", self.label));
        let mut image = self.base.clone();
        let mut tally = Tally {
            total: self.ops.len() + 1,
            clean: 0,
            loud: Vec::new(),
            silent: Vec::new(),
        };

        for k in 0..=self.ops.len() {
            if k > 0 {
                Self::apply(&mut image, &self.ops[k - 1]);
            }
            std::fs::write(&path, &image).unwrap();
            let after = if k == 0 {
                "the starting file".to_string()
            } else {
                self.ops[k - 1].describe()
            };
            match check(&path) {
                Verdict::Clean => tally.clean += 1,
                Verdict::Loud(why) => tally
                    .loud
                    .push(std::format!("after op {k} ({after}): {why}")),
                Verdict::Silent(why) => tally
                    .silent
                    .push(std::format!("after op {k} ({after}): {why}")),
            }
        }
        std::fs::remove_file(&path).ok();
        // How wide the sweep actually was, for whoever is changing it. The
        // assertions below are rules rather than counts — an exact figure here is
        // a property of one platform's allocator and page size — so the numbers
        // are printed on request instead of pinned.
        if std::env::var_os("CRASH_REPLAY_STATS").is_some() {
            std::eprintln!(
                "{}: {} prefixes, {} clean, {} loud, {} silent",
                self.label,
                tally.total,
                tally.clean,
                tally.loud.len(),
                tally.silent.len()
            );
        }
        tally.assert_sound(&self.label, min_clean);
        tally
    }
}

/// What one sweep found, kept so a caller can compare two of them.
struct Tally {
    total: usize,
    clean: usize,
    /// Why each prefix refused to read. The benign outcome, but the reasons are
    /// what say *which* benign outcome, and they are the first thing to read when
    /// the floor below fails.
    loud: Vec<String>,
    silent: Vec<String>,
}

impl Tally {
    /// Fail on any silent corruption, and on a run too quiet to have proved
    /// anything.
    fn assert_sound(&self, label: &str, min_clean: f64) {
        assert!(
            self.silent.is_empty(),
            "{label}: {} of {} replayed prefixes read cleanly and returned the wrong data:\n  {}",
            self.silent.len(),
            self.total,
            self.silent.join("\n  ")
        );
        // Without this a checker that called every state loud, or a workload that
        // quietly stopped doing work, would report no silent corruption and pass.
        assert!(
            self.clean as f64 >= min_clean * self.total as f64,
            "{label}: only {} of {} prefixes read cleanly, below the {min_clean} floor, \
             so this run proved nothing. The {} that refused:\n  {}",
            self.clean,
            self.total,
            self.loud.len(),
            self.loud.join("\n  ")
        );
    }
}

// ---------------------------------------------------------------------------
// The append workload
// ---------------------------------------------------------------------------

/// Elements the base file already holds when recording starts.
const WARMUP: i32 = 270;
/// Elements each recorded round appends.
const ROUND: i32 = 4;
/// Recorded rounds.
const ROUNDS: i32 = 70;
/// Chunk length, small enough that the warm-up crosses an Extensible-Array data
/// block *and* the super-block level above it rather than staying in the index
/// block's inline elements.
const CHUNK: u64 = 4;

/// Build a chunked, unlimited `i32` dataset holding `0..n`, and append to it
/// until it holds `WARMUP`. The appends are what matter: a file *written* with
/// 270 elements has one index shape, and a file *appended* to 270 has the shape
/// this harness is about, with the blocks allocated in the order an append
/// allocates them.
fn warmed_base(path: &Path, paged: bool) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096);
    }
    b.create_dataset("d")
        .with_i32_data(&[0i32])
        .with_shape(&[1])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[CHUNK]);
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = 1i32;
    while n < WARMUP {
        let take = ROUND.min(WARMUP - n);
        append(&mut s, n, take);
        n += take;
    }
    drop(s);
}

/// Append `count` elements continuing the sequence at `from`.
fn append(s: &mut WriteEngine, from: i32, count: i32) {
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + count).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path("d"), &b, 4)
        .unwrap();
}

/// Read `d` and require that whatever length the file claims, every element up to
/// it is the one that belongs there, and that the length is between the one the
/// recording started at and the one it ended at.
///
/// Element *i* holds the value *i*, so a stale byte, a hole read as a fill value,
/// or a block resurrected from a previous shape all show up as a wrong value
/// rather than having to be looked for. A file that simply has fewer elements is
/// a crash before that append published, which is the whole point of the
/// barriers.
fn appended_prefix_is_intact(path: &Path, lo: i32, hi: i32) -> Verdict {
    let read = crate::reader::File::open(path).and_then(|f| f.dataset("d")?.read_i32());
    Verdict::of(read, |data| {
        let n = data.len() as i32;
        if !(lo..=hi).contains(&n) {
            return Err(std::format!(
                "length {n} is outside the {lo}..={hi} this crash point can produce"
            ));
        }
        if let Some(i) = (0..data.len()).find(|&i| data[i] != i as i32) {
            return Err(std::format!(
                "element {i} is {} rather than {i} (length {n})",
                data[i]
            ));
        }
        Ok(())
    })
}

/// Stopping an in-place append at *any* write must leave a file that either
/// refuses to open or reads as an intact prefix of the sequence.
///
/// This is the test the two barriers in [`crate::chunk_index_inplace`] were added
/// for. Deleting either one makes it fail here rather than in a later session:
/// the index block or data block pointer is published while the block itself is
/// still only in the buffer, so the next read follows it past end-of-file.
#[test]
fn appending_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append.h5");
    warmed_base(&path, false);

    let rec = Recording::of("append", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(dir.path(), 0.5, |p| {
        appended_prefix_is_intact(p, WARMUP, hi)
    });
}

/// The same sweep on a paged file, where the free-space managers are written and
/// re-homed at close and the allocator rounds to a page. That teardown is a
/// second set of low-address publish points, at the end of a session rather than
/// of an operation.
#[test]
fn appending_to_a_paged_file_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_paged.h5");
    warmed_base(&path, true);

    let rec = Recording::of("append-paged", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        s.finalize_persist().unwrap();
        drop(s);
    });

    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(dir.path(), 0.5, |p| {
        appended_prefix_is_intact(p, WARMUP, hi)
    });
}

/// A file left by a crash must not merely *read* — it must be usable. Every
/// prefix that reads cleanly is reopened, appended to, and read again, which is
/// what catches a state that decodes but whose index no longer describes where
/// the next element goes.
///
/// Fewer rounds than the sweeps above: each prefix costs a session here rather
/// than a read.
#[test]
fn a_crashed_append_can_be_reopened_and_appended_to() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recover.h5");
    warmed_base(&path, false);

    const RECOVER_ROUNDS: i32 = 8;
    let rec = Recording::of("recover", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..RECOVER_ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    let hi = WARMUP + RECOVER_ROUNDS * ROUND;
    rec.replay_every_prefix(dir.path(), 0.5, |p| {
        // Read it first: a file that will not open is loud, and there is nothing
        // to recover.
        match appended_prefix_is_intact(p, WARMUP, hi) {
            Verdict::Clean => {}
            other => return other,
        }
        let before = crate::reader::File::open(p)
            .and_then(|f| f.dataset("d")?.read_i32())
            .expect("just read it")
            .len() as i32;
        let reopened = (|| -> Result<(), Error> {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled)?;
            s.set_sync_policy(SyncPolicy::OnClose);
            let mut b = AppendBuilder::new();
            b.append_i32(&(before..before + ROUND).collect::<Vec<_>>());
            s.append_inplace_gathered(AppendTarget::Path("d"), &b, 4)?;
            drop(s);
            Ok(())
        })();
        if let Err(e) = reopened {
            return Verdict::Loud(std::format!("reopen and append: {e:?}"));
        }
        appended_prefix_is_intact(p, before + ROUND, before + ROUND)
    });
}

// ---------------------------------------------------------------------------
// The commit workload
// ---------------------------------------------------------------------------

/// Churn rounds in one recorded commit session.
const CHURN: i32 = 6;
/// Elements `d` starts the commit workload with, and gains per round.
const COMMIT_BASE: i32 = 64;
const COMMIT_STEP: i32 = 16;

/// Elements of `added{r}`, and of the `doomed{r}` it replaces. Different lengths
/// so a round that resurrected the wrong object reads as the wrong length rather
/// than as plausible data.
fn added_values(r: i32) -> Vec<i32> {
    (0..48 + r).map(|i| i * 7 + r).collect()
}
fn doomed_values(r: i32) -> Vec<i32> {
    (0..32 + r).map(|i| i * 3 + r).collect()
}

/// Build the file the commit workload churns: one growable dataset, and one
/// `doomed{r}` per round for the delete half to consume.
fn commit_base(path: &Path, paged: bool) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096);
    }
    b.create_dataset("d")
        .with_i32_data(&(0..COMMIT_BASE).collect::<Vec<i32>>())
        .with_shape(&[COMMIT_BASE as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[COMMIT_STEP as u64]);
    for r in 0..CHURN {
        let v = doomed_values(r);
        b.create_dataset(&std::format!("doomed{r}"))
            .with_i32_data(&v)
            .with_shape(&[v.len() as u64]);
    }
    b.write(path).unwrap();
}

/// `d`'s elements, then each round's `added{r}` and `doomed{r}` — `None` where
/// the path is not in the file at all.
type CommitState = (Vec<i32>, Vec<Option<Vec<i32>>>, Vec<Option<Vec<i32>>>);

/// A commit publishes at its superblock write, so every object must read as
/// either its pre-commit or its post-commit state — never a mixture, and never
/// something that is neither.
///
/// Each round creates one object and deletes another, so a later round reuses
/// space an earlier one freed (issue #261). That is what makes "reads as
/// something that is neither" reachable rather than theoretical: a header
/// published over a region a previous commit released is a *readable* file whose
/// dataset returns the deleted object's bytes.
fn commit_state_is_one_or_the_other(path: &Path) -> Verdict {
    let read = (|| -> Result<CommitState, Error> {
        let f = crate::reader::File::open(path)?;
        let d = f.dataset("d")?.read_i32()?;
        // A path that is absent is a state, not a failure; a path that is present
        // and unreadable is a failure, and stays an `Err`.
        let mut added = Vec::new();
        let mut doomed = Vec::new();
        for r in 0..CHURN {
            added.push(match f.dataset(&std::format!("added{r}")) {
                Ok(ds) => Some(ds.read_i32()?),
                Err(_) => None,
            });
            doomed.push(match f.dataset(&std::format!("doomed{r}")) {
                Ok(ds) => Some(ds.read_i32()?),
                Err(_) => None,
            });
        }
        Ok((d, added, doomed))
    })();

    Verdict::of(read, |(d, added, doomed)| {
        // `d` is judged on its own. An in-place append publishes at its own
        // phase 4, before the staged commit in the same round runs, so its length
        // says nothing about whether that commit landed — the two are separate
        // publish points and this must not conflate them.
        let n = d.len() as i32;
        if !(COMMIT_BASE..=COMMIT_BASE + CHURN * COMMIT_STEP).contains(&n) {
            return Err(std::format!(
                "`d` has {n} elements, outside the {COMMIT_BASE}..={} this session can produce",
                COMMIT_BASE + CHURN * COMMIT_STEP
            ));
        }
        if let Some(i) = (0..d.len()).find(|&i| d[i] != i as i32) {
            return Err(std::format!("`d`[{i}] is {} rather than {i}", d[i]));
        }

        for r in 0..CHURN {
            if let Some(a) = &added[r as usize] {
                if *a != added_values(r) {
                    return Err(std::format!(
                        "`added{r}` is present with the wrong bytes (len {}, wanted {})",
                        a.len(),
                        added_values(r).len()
                    ));
                }
            }
            if let Some(x) = &doomed[r as usize] {
                if *x != doomed_values(r) {
                    return Err(std::format!(
                        "`doomed{r}` is still present but its bytes have changed"
                    ));
                }
            }
            // The create and the delete are staged together and published by one
            // superblock write, so they land together or not at all. This is the
            // commit's atomicity, and the one thing a torn commit cannot satisfy.
            if added[r as usize].is_some() != doomed[r as usize].is_none() {
                return Err(std::format!(
                    "round {r} is half-committed: added{r} {}, doomed{r} {}",
                    if added[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                    if doomed[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    }
                ));
            }
        }
        // Rounds commit in order, so the committed ones are a prefix. A later
        // round present while an earlier one is missing means a commit was
        // published over a state it did not build on.
        let committed: Vec<bool> = (0..CHURN).map(|r| added[r as usize].is_some()).collect();
        if let Some(r) = (1..CHURN as usize).find(|&r| committed[r] && !committed[r - 1]) {
            return Err(std::format!(
                "round {r} committed but round {} did not, though {r} came second",
                r - 1
            ));
        }
        Ok(())
    })
}

/// One recorded churn round: grow a dataset in place, create another, delete a
/// third, and commit. The three edits take different routes to the disk — the
/// append patches the header in place, the create and the delete both go through
/// the staged commit — so one recording covers all of them.
fn churn(s: &mut WriteEngine, r: i32) {
    let from = COMMIT_BASE + r * COMMIT_STEP;
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + COMMIT_STEP).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path("d"), &b, 4)
        .unwrap();

    let v = added_values(r);
    let mut db = DatasetBuilder::new(&std::format!("added{r}"));
    db.with_i32_data(&v).with_shape(&[v.len() as u64]);
    s.stage_created_dataset(&std::format!("/added{r}"), db)
        .unwrap();
    s.delete(&std::format!("doomed{r}")).unwrap();
    s.commit().unwrap();

    // The workload has to have done all three, or the sweep is judging a session
    // that quietly did nothing.
    assert!(!s.has_staged_edits(), "the commit left edits staged");
}

/// The staged-commit path, swept over both backings and both file-space
/// strategies. A commit's publish point is the superblock at address zero, which
/// is the lowest address in the file and therefore the first thing gathering
/// wants to issue.
#[test]
fn committing_survives_a_crash_at_every_write() {
    for paged in [false, true] {
        for bounded in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("commit.h5");
            commit_base(&path, paged);
            let label = std::format!(
                "commit-{}-{}",
                if paged { "paged" } else { "plain" },
                if bounded { "bounded" } else { "mirrored" }
            );

            let rec = Recording::of(&label, &path, |p| {
                let mut s = if bounded {
                    WriteEngine::open_rw_with_strategy(
                        p,
                        MetadataCacheConfig::new(64 * 1024),
                        FileLocking::Enabled,
                        MemoryStrategy::Bounded,
                    )
                    .unwrap()
                } else {
                    WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap()
                };
                s.set_sync_policy(SyncPolicy::OnClose);
                for r in 0..CHURN {
                    churn(&mut s, r);
                }
                if paged {
                    s.finalize_persist().unwrap();
                }
                drop(s);
            });

            rec.replay_every_prefix(dir.path(), 0.5, commit_state_is_one_or_the_other);
        }
    }
}

// ---------------------------------------------------------------------------
// Gathering against not gathering
// ---------------------------------------------------------------------------

/// Gathering makes each individual write *larger*, which sounds like it should
/// widen the window a crash can land in. It does the opposite, and this measures
/// by how much.
///
/// The reason is that the format's publish points are frequently a pair of
/// writes — a value, and the checksum covering it — and a pair is atomic only
/// when both land together. Straight-through writing never joins them.
/// Gathering joins them whenever they share a page, which is every object header
/// this crate's own writer produces below about 3.9 KB (the residue above that
/// size is issue #307).
///
/// So the same workload, swept the same way, leaves *fewer* unreadable states
/// when gathered. Both are run here in one process against one warmed base file,
/// so the comparison is between the two modes and not between two machines.
#[test]
fn gathering_leaves_fewer_unreadable_crash_states_than_not_gathering() {
    use crate::image::WriteBuffering;

    const ROUNDS_EACH: i32 = 20;
    let hi = WARMUP + ROUNDS_EACH * ROUND;

    let sweep = |label: &str, mode: Option<WriteBuffering>| -> Tally {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compare.h5");
        warmed_base(&path, false);
        let rec = Recording::of(label, &path, |p| {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if let Some(mode) = mode {
                s.set_write_buffering(mode).unwrap();
            }
            for r in 0..ROUNDS_EACH {
                append(&mut s, WARMUP + r * ROUND, ROUND);
            }
            drop(s);
        });
        rec.replay_every_prefix(dir.path(), 0.5, |p| {
            appended_prefix_is_intact(p, WARMUP, hi)
        })
    };

    // `None` keeps the session default, which is the gathering a locked
    // read-write session takes.
    let gathered = sweep("compare-gathered", None);
    let straight = sweep("compare-unbuffered", Some(WriteBuffering::Unbuffered));

    // Neither mode may corrupt silently — `assert_sound` has already said so for
    // both, and that is the guarantee. This is the softer, measured claim on top
    // of it.
    assert!(
        gathered.loud.len() < straight.loud.len(),
        "gathering was supposed to leave fewer unreadable crash states, but left \
         {} of {} against {} of {}. The gathered ones:\n  {}",
        gathered.loud.len(),
        gathered.total,
        straight.loud.len(),
        straight.total,
        gathered.loud.join("\n  ")
    );
    // And the reason it is fewer: the states straight-through writing leaves are
    // torn publish points, a value written without the checksum that covers it.
    let torn = straight
        .loud
        .iter()
        .filter(|why| why.contains("ChecksumMismatch"))
        .count();
    assert!(
        torn > 0,
        "expected the unbuffered sweep to tear a checksum away from the value it \
         covers, but none of its {} unreadable states says so:\n  {}",
        straight.loud.len(),
        straight.loud.join("\n  ")
    );
}
