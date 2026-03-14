//! SHINES provenance: SHA-256 content hashing, provenance attributes, and
//! data-integrity verification.
//!
//! Enable with the `provenance` Cargo feature (on by default).

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use sha2::{Digest, Sha256};

use crate::attribute::AttributeMessage;
use crate::data_layout::DataLayout;
use crate::data_read::read_raw_data;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::object_header::ObjectHeader;
use crate::type_builders::{build_attr_message, AttrValue};

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

// ---- Verification ----

/// Result of a provenance verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// Hash matches — data integrity confirmed.
    Ok,
    /// Hash mismatch — stored vs computed.
    Mismatch { stored: String, computed: String },
    /// No provenance hash attribute found on this dataset.
    NoHash,
}

/// Verify the integrity of a dataset by recomputing its SHA-256 hash and
/// comparing it against the stored `_provenance_sha256` attribute.
///
/// `file_data` is the entire HDF5 file bytes; `header` is the parsed object
/// header for the dataset of interest.
pub fn verify_dataset(
    file_data: &[u8],
    header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<VerifyResult, FormatError> {
    // 1. Extract all attributes (compact + dense).
    let attrs = crate::attribute::extract_attributes_full(
        file_data,
        header,
        offset_size,
        length_size,
    )?;

    // 2. Find the stored hash.
    let stored_hash = attrs
        .iter()
        .find(|a| a.name == ATTR_SHA256)
        .and_then(|a| core::str::from_utf8(&a.raw_data).ok())
        .map(|s| s.trim_end_matches('\0').to_string());

    let stored_hash = match stored_hash {
        Some(h) => h,
        None => return Ok(VerifyResult::NoHash),
    };

    // 3. Read the raw dataset data.
    let dt_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == crate::message_type::MessageType::Datatype)
        .ok_or_else(|| FormatError::SerializationError("missing Datatype message".into()))?;
    let ds_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == crate::message_type::MessageType::Dataspace)
        .ok_or_else(|| FormatError::SerializationError("missing Dataspace message".into()))?;
    let dl_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == crate::message_type::MessageType::DataLayout)
        .ok_or_else(|| FormatError::SerializationError("missing DataLayout message".into()))?;

    let (dt, _) = Datatype::parse(&dt_msg.data)?;
    let ds = Dataspace::parse(&ds_msg.data, length_size)?;
    let dl = DataLayout::parse(&dl_msg.data, offset_size, length_size)?;

    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == crate::message_type::MessageType::FilterPipeline)
        .map(|m| crate::filter_pipeline::FilterPipeline::parse(&m.data))
        .transpose()?;

    let raw = match &dl {
        DataLayout::Chunked { .. } => crate::chunked_read::read_chunked_data(
            file_data,
            &dl,
            &ds,
            &dt,
            pipeline.as_ref(),
            offset_size,
            length_size,
        )?,
        _ => read_raw_data(file_data, &dl, &ds, &dt)?,
    };

    // 4. Compare.
    let computed = sha256_hex(&raw);
    if computed == stored_hash {
        Ok(VerifyResult::Ok)
    } else {
        Ok(VerifyResult::Mismatch {
            stored: stored_hash,
            computed,
        })
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

    #[test]
    fn roundtrip_provenance_on_file() {
        use crate::file_writer::FileWriter;
        use crate::group_v2::resolve_path_any;
        use crate::signature;
        use crate::superblock::Superblock;

        let raw_data: Vec<u8> = (0..24u64)
            .flat_map(|v| (v as f64).to_le_bytes())
            .collect();
        let expected_hash = sha256_hex(&raw_data);

        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("sensor");
        ds.with_f64_data(
            &(0..24).map(|v| v as f64).collect::<Vec<_>>(),
        );
        ds.set_attr(ATTR_SHA256, AttrValue::String(expected_hash.clone()));
        ds.set_attr(
            ATTR_CREATOR,
            AttrValue::String("test-suite".into()),
        );
        ds.set_attr(
            ATTR_TIMESTAMP,
            AttrValue::String("2026-02-19T12:00:00Z".into()),
        );
        let bytes = fw.finish().unwrap();

        // Verify round-trip
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "sensor").unwrap();
        let hdr = crate::object_header::ObjectHeader::parse(
            &bytes,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();

        let result = verify_dataset(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(result, VerifyResult::Ok);
    }

    #[test]
    fn verify_no_hash_attribute() {
        use crate::file_writer::FileWriter;
        use crate::group_v2::resolve_path_any;
        use crate::signature;
        use crate::superblock::Superblock;

        let mut fw = FileWriter::new();
        fw.create_dataset("plain").with_f64_data(&[1.0, 2.0]);
        let bytes = fw.finish().unwrap();

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "plain").unwrap();
        let hdr = crate::object_header::ObjectHeader::parse(
            &bytes,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();

        let result = verify_dataset(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(result, VerifyResult::NoHash);
    }
}
