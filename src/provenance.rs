//! SHINES provenance: SHA-256 content hashing, provenance attributes, and
//! data-integrity verification.
//!
//! Enable with the `provenance` Cargo feature (on by default).

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use sha2::{Digest, Sha256};

use crate::attribute::AttributeMessage;
use crate::type_builders::{AttrValue, build_attr_message};

// ---- Attribute name constants ----

/// SHA-256 hex digest of the raw dataset bytes.
pub const ATTR_SHA256: &str = "_provenance_sha256";
/// Creator identifier (tool/user).
pub const ATTR_CREATOR: &str = "_provenance_creator";
/// ISO-8601 timestamp when the dataset was written.
pub const ATTR_TIMESTAMP: &str = "_provenance_timestamp";
/// Optional free-form description of the data source.
pub const ATTR_SOURCE: &str = "_provenance_source";

// ---- SHA-256 hashing ----

/// Compute the SHA-256 digest of `data` and return the lowercase hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let mut hex = String::with_capacity(64);
    for byte in hash.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

// ---- Provenance metadata builder ----

/// Collects provenance information to be stored as HDF5 attributes.
pub struct Provenance {
    pub creator: String,
    pub timestamp: String,
    pub source: Option<String>,
}

impl Provenance {
    /// Build provenance attribute messages for the given raw dataset bytes.
    ///
    /// Returns a `Vec<AttributeMessage>` containing:
    /// - `_provenance_sha256`   — hex digest of `raw_data`
    /// - `_provenance_creator`  — the creator string
    /// - `_provenance_timestamp` — the timestamp string
    /// - `_provenance_source`   — (optional) source description
    pub fn build_attrs(&self, raw_data: &[u8]) -> Vec<AttributeMessage> {
        let hash = sha256_hex(raw_data);
        let mut attrs = Vec::with_capacity(4);
        let hash_val = AttrValue::String(hash);
        attrs.push(build_attr_message(ATTR_SHA256, &hash_val));
        let creator_val = AttrValue::String(self.creator.clone());
        attrs.push(build_attr_message(ATTR_CREATOR, &creator_val));
        let ts_val = AttrValue::String(self.timestamp.clone());
        attrs.push(build_attr_message(ATTR_TIMESTAMP, &ts_val));
        if let Some(ref src) = self.source {
            let src_val = AttrValue::String(src.clone());
            attrs.push(build_attr_message(ATTR_SOURCE, &src_val));
        }
        attrs
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_empty() {
        // Well-known: SHA-256 of empty input
        let h = sha256_hex(b"");
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_hello() {
        let h = sha256_hex(b"hello");
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn provenance_builds_attrs_without_source() {
        let prov = Provenance {
            creator: "rustyhdf5".into(),
            timestamp: "2026-02-19T00:00:00Z".into(),
            source: None,
        };
        let attrs = prov.build_attrs(b"hello");
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, ATTR_SHA256);
        assert_eq!(attrs[1].name, ATTR_CREATOR);
        assert_eq!(attrs[2].name, ATTR_TIMESTAMP);
    }

    #[test]
    fn provenance_builds_attrs_with_source() {
        let prov = Provenance {
            creator: "test".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            source: Some("sensor_42".into()),
        };
        let attrs = prov.build_attrs(b"data");
        assert_eq!(attrs.len(), 4);
        assert_eq!(attrs[3].name, ATTR_SOURCE);
    }
}
