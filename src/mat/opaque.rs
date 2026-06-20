//! Public Rust views over decoded MATLAB MCOS opaque value classes.
//!
//! When [`crate::mat::from_bytes`] reads a MATLAB `datetime`, `duration`, or
//! `categorical` variable, it decodes the hidden `#subsystem#/MCOS` object into
//! its logical components. These structs deserialize from that decoded form, so
//! a field of one of these types in your `#[derive(Deserialize)]` struct picks
//! up the corresponding MATLAB variable directly:
//!
//! ```no_run
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat::{self, MatDatetime, MatCategorical};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Log {
//!     timestamps: MatDatetime,
//!     labels: MatCategorical,
//! }
//!
//! let log: Log = mat::from_bytes(&std::fs::read("log.mat").unwrap()).unwrap();
//! let epoch_ns = log.timestamps.nanoseconds();
//! let resolved = log.labels.labels(); // Vec<Option<String>>
//! # let _ = (epoch_ns, resolved);
//! # }
//! ```
//!
//! The representations are lossless: the raw MATLAB storage (milliseconds since
//! the Unix epoch, category codes, …) is preserved, and helper methods offer the
//! common derived views.

use serde::Deserialize;

/// A decoded MATLAB `datetime` array.
///
/// MATLAB stores datetimes as milliseconds since the Unix epoch
/// (1970-01-01 UTC, with no time-zone offset folded in). [`millis_utc`] holds
/// the whole-millisecond instant of each element and [`sub_ms`] a
/// sub-millisecond correction, both preserved losslessly. [`tz`] and [`fmt`] are
/// display metadata and do not shift the stored instant.
///
/// `NaT` (not-a-time) elements appear as `NaN` in [`millis_utc`].
///
/// [`millis_utc`]: MatDatetime::millis_utc
/// [`sub_ms`]: MatDatetime::sub_ms
/// [`tz`]: MatDatetime::tz
/// [`fmt`]: MatDatetime::fmt
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MatDatetime {
    /// Milliseconds since 1970-01-01 UTC, one per element (row-major order).
    pub millis_utc: Vec<f64>,
    /// Sub-millisecond correction per element, paired with `millis_utc`.
    #[serde(default)]
    pub sub_ms: Vec<f64>,
    /// Time-zone name when the datetime is zoned (display metadata only).
    #[serde(default)]
    pub tz: Option<String>,
    /// Display format string, when present.
    #[serde(default)]
    pub fmt: Option<String>,
}

impl MatDatetime {
    /// Nanoseconds since the Unix epoch for each element, combining the whole-
    /// and sub-millisecond parts.
    ///
    /// The sub-millisecond part is combined as `(millis_utc + sub_ms) * 1e6`.
    /// The exact scale of the stored sub-millisecond component has not been
    /// pinned against real MATLAB output yet; for whole-millisecond datetimes
    /// (where `sub_ms` is zero) the result is exact regardless.
    #[must_use]
    pub fn nanoseconds(&self) -> Vec<f64> {
        self.millis_utc
            .iter()
            .enumerate()
            .map(|(i, &millis)| {
                let sub = self.sub_ms.get(i).copied().unwrap_or(0.0);
                (millis + sub) * 1e6
            })
            .collect()
    }
}

/// A decoded MATLAB `duration` array.
///
/// MATLAB stores durations as milliseconds. [`millis`] is preserved losslessly;
/// [`fmt`] is the display unit and does not change the stored value.
///
/// [`millis`]: MatDuration::millis
/// [`fmt`]: MatDuration::fmt
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MatDuration {
    /// Length of each element in milliseconds (row-major order).
    pub millis: Vec<f64>,
    /// Display unit (`s`/`m`/`h`/`d`/`y`), when present.
    #[serde(default)]
    pub fmt: Option<String>,
}

impl MatDuration {
    /// Each element's length in seconds.
    #[must_use]
    pub fn seconds(&self) -> Vec<f64> {
        self.millis.iter().map(|&m| m / 1000.0).collect()
    }
}

/// A decoded MATLAB `categorical` array.
///
/// [`codes`] holds the 1-based category index of each element (`0` is
/// `<undefined>`); [`categories`] is the category-name pool. [`is_ordinal`] and
/// [`is_protected`] mirror the MATLAB flags.
///
/// [`codes`]: MatCategorical::codes
/// [`categories`]: MatCategorical::categories
/// [`is_ordinal`]: MatCategorical::is_ordinal
/// [`is_protected`]: MatCategorical::is_protected
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MatCategorical {
    /// 1-based category index per element (`0` = `<undefined>`), row-major.
    pub codes: Vec<u32>,
    /// The category-name pool, indexed by `code - 1`.
    pub categories: Vec<String>,
    /// Whether the categories are ordered (an ordinal categorical).
    #[serde(default)]
    pub is_ordinal: bool,
    /// Whether the category set is protected (no implicit new categories).
    #[serde(default)]
    pub is_protected: bool,
}

impl MatCategorical {
    /// The category label of each element, `None` for `<undefined>` (code `0`)
    /// or an out-of-range code.
    #[must_use]
    pub fn labels(&self) -> Vec<Option<String>> {
        self.codes
            .iter()
            .map(|&code| {
                if code == 0 {
                    None
                } else {
                    self.categories.get((code - 1) as usize).cloned()
                }
            })
            .collect()
    }
}
