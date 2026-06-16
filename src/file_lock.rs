//! OS advisory file locking for the write/edit/read open paths (issue #73).
//!
//! This is the crash-safe half of HDF5's two-layer concurrency model, and the
//! `hdf5-pure` analogue of `H5Pset_file_locking` / the `HDF5_USE_FILE_LOCKING`
//! environment variable. It is deliberately distinct from the *superblock
//! consistency flag* (the durable `status_flags` byte that a SWMR writer sets;
//! see [`crate::SwmrWriter`]):
//!
//! - An **advisory lock** is owned by the kernel and tied to the open file. It
//!   is released automatically when the process exits *for any reason* — clean
//!   exit, panic, `SIGKILL`, even power loss — so it never leaves stale state
//!   and is the authoritative signal for "a writer is alive *right now*".
//! - The **on-disk flag** is just a byte; only userspace code at clean shutdown
//!   can reset it, so a crash freezes it set. That is why it cannot be the
//!   liveness signal, and why a separate recovery step
//!   ([`crate::SwmrWriter::clear_swmr_flag`], the `h5clear -s` equivalent)
//!   exists.
//!
//! Writers ([`crate::SwmrWriter`], [`crate::EditSession`]) take an **exclusive**
//! lock; non-SWMR readers ([`crate::File::open`] and friends) take a **shared**
//! lock held only as long as they hold the handle. SWMR readers
//! ([`crate::File::open_swmr`]) take *no* lock on purpose — opting into reading a
//! file under concurrent write is the whole point of SWMR, and a shared lock
//! would conflict with the writer's exclusive one.
//!
//! Locking uses the cross-platform [`std::fs::File`] lock API (Unix `flock`,
//! Windows `LockFileEx`), so it adds no dependency, and it lives only in the
//! already `std`-gated open paths, so `no_std`/`wasm` builds are unaffected.

use std::fs::{File, TryLockError};
use std::path::Path;

use crate::error::Error;

/// Policy for OS advisory file locking when opening a file.
///
/// The default is [`FileLocking::Enabled`]. The `HDF5_USE_FILE_LOCKING`
/// environment variable, when set to a recognized value, overrides the
/// requested policy (matching the reference HDF5 library): `FALSE`/`0`/`NO`/`OFF`
/// disable locking, `BEST_EFFORT` selects best-effort, and `TRUE`/`1`/`YES`/`ON`
/// enable it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileLocking {
    /// Acquire the lock, and fail the open with [`Error::Io`] if the filesystem
    /// does not support locking. A lock held by another process always fails
    /// the open with [`Error::FileLocked`].
    #[default]
    Enabled,
    /// Do not attempt to lock the file at all.
    Disabled,
    /// Attempt to lock, but proceed *without* a lock when the filesystem reports
    /// that locking is unavailable (e.g. some NFS / network mounts). A lock that
    /// is genuinely *held* by another process still fails the open. Mirrors the
    /// reference library's `BEST_EFFORT` / `ignore_disabled_locks`.
    BestEffort,
}

/// Parse a recognized `HDF5_USE_FILE_LOCKING` value into a policy, or `None` for
/// an unrecognized value (in which case the requested policy is kept).
///
/// Pure (no environment access) so it can be unit-tested without the
/// process-global, edition-2024-`unsafe` env mutators.
fn parse_env(value: &str) -> Option<FileLocking> {
    let v = value.trim();
    if v.eq_ignore_ascii_case("FALSE")
        || v == "0"
        || v.eq_ignore_ascii_case("NO")
        || v.eq_ignore_ascii_case("OFF")
    {
        Some(FileLocking::Disabled)
    } else if v.eq_ignore_ascii_case("BEST_EFFORT") {
        Some(FileLocking::BestEffort)
    } else if v.eq_ignore_ascii_case("TRUE")
        || v == "1"
        || v.eq_ignore_ascii_case("YES")
        || v.eq_ignore_ascii_case("ON")
    {
        Some(FileLocking::Enabled)
    } else {
        None
    }
}

/// Apply the `HDF5_USE_FILE_LOCKING` environment override to a requested policy.
/// The environment variable, when set to a recognized value, takes precedence.
fn resolve(requested: FileLocking) -> FileLocking {
    std::env::var("HDF5_USE_FILE_LOCKING")
        .ok()
        .and_then(|v| parse_env(&v))
        .unwrap_or(requested)
}

/// Acquire an **exclusive** advisory lock on `handle` for a writer open.
///
/// Non-blocking: if another process holds a conflicting (shared or exclusive)
/// lock, this returns [`Error::FileLocked`] immediately rather than waiting.
pub(crate) fn acquire_exclusive(
    handle: &File,
    requested: FileLocking,
    path: &Path,
) -> Result<(), Error> {
    acquire(handle, requested, path, false)
}

/// Acquire a **shared** advisory lock on `handle` for a reader open. Fails with
/// [`Error::FileLocked`] only if a writer currently holds the exclusive lock.
pub(crate) fn acquire_shared(
    handle: &File,
    requested: FileLocking,
    path: &Path,
) -> Result<(), Error> {
    acquire(handle, requested, path, true)
}

fn acquire(handle: &File, requested: FileLocking, path: &Path, shared: bool) -> Result<(), Error> {
    let mode = resolve(requested);
    if mode == FileLocking::Disabled {
        return Ok(());
    }
    let result = if shared {
        handle.try_lock_shared()
    } else {
        handle.try_lock()
    };
    match result {
        Ok(()) => Ok(()),
        // A conflicting lock is genuinely held by another process: the file is
        // in use. `BestEffort` does not soften this — only *unavailable*
        // locking is tolerated, not active contention.
        Err(TryLockError::WouldBlock) => Err(Error::FileLocked(format!(
            "{}: file is already locked by another process. If a previous writer \
             crashed, the OS lock is released automatically (try again); a leftover \
             on-disk SWMR flag can be cleared with SwmrWriter::clear_swmr_flag. Set \
             HDF5_USE_FILE_LOCKING=FALSE or pass FileLocking::Disabled to bypass locking.",
            path.display(),
        ))),
        // Locking failed for another reason — typically the filesystem does not
        // support advisory locks (some NFS / network mounts).
        Err(TryLockError::Error(e)) => match mode {
            FileLocking::BestEffort => Ok(()),
            _ => Err(Error::Io(e)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_recognizes_disable_values() {
        for v in ["FALSE", "false", "0", "No", "off", " false "] {
            assert_eq!(parse_env(v), Some(FileLocking::Disabled), "value {v:?}");
        }
    }

    #[test]
    fn parse_env_recognizes_enable_and_best_effort() {
        for v in ["TRUE", "true", "1", "Yes", "on"] {
            assert_eq!(parse_env(v), Some(FileLocking::Enabled), "value {v:?}");
        }
        assert_eq!(parse_env("BEST_EFFORT"), Some(FileLocking::BestEffort));
        assert_eq!(parse_env("best_effort"), Some(FileLocking::BestEffort));
    }

    #[test]
    fn parse_env_unrecognized_is_none() {
        assert_eq!(parse_env(""), None);
        assert_eq!(parse_env("maybe"), None);
        assert_eq!(parse_env("2"), None);
    }

    #[test]
    fn default_is_enabled() {
        assert_eq!(FileLocking::default(), FileLocking::Enabled);
    }
}
