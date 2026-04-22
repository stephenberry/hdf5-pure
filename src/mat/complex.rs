//! Complex number newtypes recognized by the MAT serializer.
//!
//! These use a sentinel struct name so the generic MAT serializer can detect
//! complex values and write them as HDF5 compound `{real, imag}` datasets,
//! which is how MATLAB stores complex arrays in v7.3 files.
//!
//! For `Vec<Complex64>` the serializer produces a compound dataset of shape
//! `[1, N]`; a bare `Complex64` becomes a compound scalar of shape `[1, 1]`.

use core::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(crate) const COMPLEX32_SENTINEL: &str = "__hdf5_pure_mat_Complex32__";
pub(crate) const COMPLEX64_SENTINEL: &str = "__hdf5_pure_mat_Complex64__";

macro_rules! complex_type {
    ($name:ident, $scalar:ty, $sentinel:ident) => {
        /// Complex number stored as a MATLAB-compatible compound dataset.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name {
            /// Real part.
            pub re: $scalar,
            /// Imaginary part.
            pub im: $scalar,
        }

        impl $name {
            /// Build a new complex value.
            pub const fn new(re: $scalar, im: $scalar) -> Self {
                Self { re, im }
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut s = serializer.serialize_struct($sentinel, 2)?;
                s.serialize_field("real", &self.re)?;
                s.serialize_field("imag", &self.im)?;
                s.end()
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct ComplexVisitor;

                impl<'de> Visitor<'de> for ComplexVisitor {
                    type Value = $name;
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str(concat!(
                            stringify!($name),
                            " struct with fields `real` and `imag`"
                        ))
                    }
                    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<$name, A::Error> {
                        let mut re: Option<$scalar> = None;
                        let mut im: Option<$scalar> = None;
                        while let Some(key) = map.next_key::<String>()? {
                            match key.as_str() {
                                "real" => re = Some(map.next_value()?),
                                "imag" => im = Some(map.next_value()?),
                                _ => {
                                    let _: serde::de::IgnoredAny = map.next_value()?;
                                }
                            }
                        }
                        Ok($name {
                            re: re.ok_or_else(|| de::Error::missing_field("real"))?,
                            im: im.ok_or_else(|| de::Error::missing_field("imag"))?,
                        })
                    }
                }

                deserializer.deserialize_struct($sentinel, &["real", "imag"], ComplexVisitor)
            }
        }
    };
}

complex_type!(Complex32, f32, COMPLEX32_SENTINEL);
complex_type!(Complex64, f64, COMPLEX64_SENTINEL);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_compare() {
        let a = Complex64::new(1.0, -2.0);
        let b = Complex64 { re: 1.0, im: -2.0 };
        assert_eq!(a, b);
    }
}
