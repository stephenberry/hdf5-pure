//! HDF5 object header message type identifiers.

use core::fmt;

/// Recognized HDF5 header message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Nil,
    Dataspace,
    LinkInfo,
    Datatype,
    FillValueOld,
    FillValue,
    Link,
    DataLayout,
    GroupInfo,
    FilterPipeline,
    Attribute,
    ObjectHeaderContinuation,
    SymbolTable,
    ObjectModificationTime,
    BTreeKValues,
    SharedMessageTable,
    AttributeInfo,
    ObjectReferenceCount,
    FileSpaceInfo,
    /// Unknown message type with its raw type ID.
    Unknown(u16),
}

impl MessageType {
    /// Convert a raw u16 type ID to a `MessageType`.
    pub fn from_u16(val: u16) -> MessageType {
        match val {
            0x0000 => MessageType::Nil,
            0x0001 => MessageType::Dataspace,
            0x0002 => MessageType::LinkInfo,
            0x0003 => MessageType::Datatype,
            0x0004 => MessageType::FillValueOld,
            0x0005 => MessageType::FillValue,
            0x0006 => MessageType::Link,
            0x0008 => MessageType::DataLayout,
            0x000A => MessageType::GroupInfo,
            0x000B => MessageType::FilterPipeline,
            0x000C => MessageType::Attribute,
            0x000F => MessageType::SharedMessageTable,
            0x0010 => MessageType::ObjectHeaderContinuation,
            0x0011 => MessageType::SymbolTable,
            0x0012 => MessageType::ObjectModificationTime,
            0x0013 => MessageType::BTreeKValues,
            0x0015 => MessageType::AttributeInfo,
            0x0016 => MessageType::ObjectReferenceCount,
            0x0017 => MessageType::FileSpaceInfo,
            other => MessageType::Unknown(other),
        }
    }

    /// Convert back to the raw u16 type ID.
    pub fn to_u16(self) -> u16 {
        match self {
            MessageType::Nil => 0x0000,
            MessageType::Dataspace => 0x0001,
            MessageType::LinkInfo => 0x0002,
            MessageType::Datatype => 0x0003,
            MessageType::FillValueOld => 0x0004,
            MessageType::FillValue => 0x0005,
            MessageType::Link => 0x0006,
            MessageType::DataLayout => 0x0008,
            MessageType::GroupInfo => 0x000A,
            MessageType::FilterPipeline => 0x000B,
            MessageType::Attribute => 0x000C,
            MessageType::SharedMessageTable => 0x000F,
            MessageType::ObjectHeaderContinuation => 0x0010,
            MessageType::SymbolTable => 0x0011,
            MessageType::ObjectModificationTime => 0x0012,
            MessageType::BTreeKValues => 0x0013,
            MessageType::AttributeInfo => 0x0015,
            MessageType::ObjectReferenceCount => 0x0016,
            MessageType::FileSpaceInfo => 0x0017,
            MessageType::Unknown(v) => v,
        }
    }
}

impl fmt::Display for MessageType {
    /// The message name as the format specification writes it, and the raw
    /// identifier for a type this crate does not recognize.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Nil => "NIL",
            Self::Dataspace => "dataspace",
            Self::LinkInfo => "link info",
            Self::Datatype => "datatype",
            Self::FillValueOld => "fill value (old)",
            Self::FillValue => "fill value",
            Self::Link => "link",
            Self::DataLayout => "data layout",
            Self::GroupInfo => "group info",
            Self::FilterPipeline => "filter pipeline",
            Self::Attribute => "attribute",
            Self::ObjectHeaderContinuation => "object header continuation",
            Self::SymbolTable => "symbol table",
            Self::ObjectModificationTime => "object modification time",
            Self::BTreeKValues => "B-tree K values",
            Self::SharedMessageTable => "shared message table",
            Self::AttributeInfo => "attribute info",
            Self::ObjectReferenceCount => "object reference count",
            Self::FileSpaceInfo => "file space info",
            Self::Unknown(id) => return write!(f, "unknown message 0x{id:04x}"),
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_types_roundtrip() {
        let known = [
            (0x0000, MessageType::Nil),
            (0x0001, MessageType::Dataspace),
            (0x0002, MessageType::LinkInfo),
            (0x0003, MessageType::Datatype),
            (0x0004, MessageType::FillValueOld),
            (0x0005, MessageType::FillValue),
            (0x0006, MessageType::Link),
            (0x0008, MessageType::DataLayout),
            (0x000A, MessageType::GroupInfo),
            (0x000B, MessageType::FilterPipeline),
            (0x000C, MessageType::Attribute),
            (0x000F, MessageType::SharedMessageTable),
            (0x0010, MessageType::ObjectHeaderContinuation),
            (0x0011, MessageType::SymbolTable),
            (0x0012, MessageType::ObjectModificationTime),
            (0x0013, MessageType::BTreeKValues),
            (0x0015, MessageType::AttributeInfo),
            (0x0016, MessageType::ObjectReferenceCount),
            (0x0017, MessageType::FileSpaceInfo),
        ];
        for (val, expected) in &known {
            let mt = MessageType::from_u16(*val);
            assert_eq!(mt, *expected);
            assert_eq!(mt.to_u16(), *val);
        }
    }

    #[test]
    fn unknown_type() {
        let mt = MessageType::from_u16(0x00FF);
        assert_eq!(mt, MessageType::Unknown(0x00FF));
        assert_eq!(mt.to_u16(), 0x00FF);
    }

    #[test]
    fn unknown_type_zero_gap() {
        // 0x0007 is not a defined type
        let mt = MessageType::from_u16(0x0007);
        assert_eq!(mt, MessageType::Unknown(0x0007));
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;

    /// `Error::MissingMessage` quotes this, so it must read as prose rather
    /// than as a Rust variant name.
    #[test]
    fn known_messages_read_as_prose() {
        assert_eq!(MessageType::Dataspace.to_string(), "dataspace");
        assert_eq!(MessageType::LinkInfo.to_string(), "link info");
        assert_eq!(
            MessageType::ObjectHeaderContinuation.to_string(),
            "object header continuation"
        );
    }

    #[test]
    fn an_unknown_message_reports_its_raw_identifier() {
        assert_eq!(
            MessageType::Unknown(0x00ff).to_string(),
            "unknown message 0x00ff"
        );
    }

    /// A message this crate names must not fall through to the raw-identifier
    /// wording, which would hide that the type is in fact recognized.
    #[test]
    fn every_known_message_has_a_name() {
        for raw in 0x0000..=0x0017u16 {
            let message = MessageType::from_u16(raw);
            if message == MessageType::Unknown(raw) {
                continue;
            }
            let shown = message.to_string();
            assert!(!shown.starts_with("unknown message"), "{raw:#06x}: {shown}");
        }
    }
}
