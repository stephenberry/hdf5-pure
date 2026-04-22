//! Top-level serializer that enforces "the root value must be a struct".
//!
//! Each top-level field becomes a MATLAB variable in the produced `.mat`
//! file. A flat `HashMap<String, T>` is also accepted, matching
//! `scipy.io.savemat`'s dict-at-root convention.

use serde::ser::{
    self, Impossible, Serialize, SerializeMap, SerializeStruct, Serializer,
};

use crate::mat::error::MatError;

use super::emit::emit_file;
use super::value_ser::{to_value, ValueSerializer};
use crate::mat::value::MatValue;

/// Serialize `value` to MAT v7.3 file bytes (with 512-byte userblock).
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, MatError> {
    let fields = value.serialize(RootSerializer)?;
    emit_file(fields)
}

/// The root serializer. Produces `Vec<(field_name, MatValue)>`.
pub(crate) struct RootSerializer;

impl Serializer for RootSerializer {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    type SerializeSeq = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTuple = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTupleStruct = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTupleVariant = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeMap = RootMapSer;
    type SerializeStruct = RootStructSer;
    type SerializeStructVariant = Impossible<Vec<(String, MatValue)>, MatError>;

    fn serialize_bool(self, _: bool) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i8(self, _: i8) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_i16(self, _: i16) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_i32(self, _: i32) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_i64(self, _: i64) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_i128(self, _: i128) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_u8(self, _: u8) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_u16(self, _: u16) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_u32(self, _: u32) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_u64(self, _: u64) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_u128(self, _: u128) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_f32(self, _: f32) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_f64(self, _: f64) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_char(self, _: char) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_str(self, _: &str) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }
    fn serialize_bytes(self, _: &[u8]) -> Result<Self::Ok, MatError> { Err(MatError::RootMustBeStruct) }

    fn serialize_none(self) -> Result<Self::Ok, MatError> {
        // `None` at root: produce an empty file (no variables).
        Ok(Vec::new())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, MatError> {
        // Treat `()` at root as an empty workspace.
        Ok(Vec::new())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, MatError> {
        Ok(Vec::new())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, MatError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<RootMapSer, MatError> {
        Ok(RootMapSer::default())
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<RootStructSer, MatError> {
        Ok(RootStructSer::default())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, MatError> {
        Err(MatError::RootMustBeStruct)
    }
}

// ---------------------------------------------------------------------------
// RootStructSer — collects top-level named fields
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct RootStructSer {
    fields: Vec<(String, MatValue)>,
}

impl SerializeStruct for RootStructSer {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), MatError> {
        let v = to_value(value)?;
        if !matches!(v, MatValue::Omit) {
            self.fields.push((key.to_owned(), v));
        }
        Ok(())
    }

    fn end(self) -> Result<Vec<(String, MatValue)>, MatError> {
        Ok(self.fields)
    }
}

// ---------------------------------------------------------------------------
// RootMapSer — accepts HashMap<String, T> at the root
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct RootMapSer {
    fields: Vec<(String, MatValue)>,
    pending_key: Option<String>,
}

impl SerializeMap for RootMapSer {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer)?;
        match key_val {
            MatValue::String(s) => {
                self.pending_key = Some(s);
                Ok(())
            }
            other => Err(MatError::Custom(format!(
                "root map keys must be strings, got {}",
                other.kind()
            ))),
        }
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        let k = self
            .pending_key
            .take()
            .ok_or_else(|| MatError::Custom("serialize_value before serialize_key".into()))?;
        let v = to_value(value)?;
        if !matches!(v, MatValue::Omit) {
            self.fields.push((k, v));
        }
        Ok(())
    }

    fn end(self) -> Result<Vec<(String, MatValue)>, MatError> {
        Ok(self.fields)
    }
}

// Silence unused-import warnings in some builds.
#[allow(dead_code)]
fn _touch<E: ser::Error>() -> E {
    E::custom("x")
}
