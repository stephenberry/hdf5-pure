//! Complex number newtypes recognized by the MAT serializer.
//!
//! These use a sentinel struct name so the generic MAT serializer can detect
//! complex values and write them as HDF5 compound `{real, imag}` datasets,
//! which is how MATLAB stores complex arrays in v7.3 files.
//!
//! For `Vec<Complex64>` the serializer produces a compound dataset of shape
//! `[1, N]`; a bare `Complex64` becomes a compound scalar of shape `[1, 1]`.
//!
//! There is one type per component class, because the component class *is* the
//! array's MATLAB class: a [`ComplexI16`] array reports `int16`, not some
//! complex-specific class, and a capture that samples as pairs of 16-bit
//! integers stores in half the space its `single` equivalent would take.
//! Nothing is ever widened on the way in or out — a value written as
//! [`ComplexI16`] reads back only as [`ComplexI16`], so the file always says
//! what it holds.

use core::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::mat::value::ComplexTag;

macro_rules! complex_types {
    ($($name:ident => $scalar:ty, $sentinel:ident, $tag:ident, $matlab:literal),* $(,)?) => {
        $(
            #[doc = concat!(
                "Sentinel struct name for [`", stringify!($name), "`]."
            )]
            pub(crate) const $sentinel: &str =
                concat!("__hdf5_pure_mat_", stringify!($name), "__");

            #[doc = concat!(
                "A complex number with `", stringify!($scalar),
                "` components, stored as a MATLAB `", $matlab,
                "` complex array."
            )]
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
                        fn visit_map<A: MapAccess<'de>>(
                            self,
                            mut map: A,
                        ) -> Result<$name, A::Error> {
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

                    deserializer.deserialize_struct(
                        $sentinel,
                        &["real", "imag"],
                        ComplexVisitor,
                    )
                }
            }
        )*

        /// The component class a complex sentinel names, or `None` for any
        /// other struct name.
        ///
        /// This is how the (de)serializer recognizes a complex value: the
        /// sentinel carries the component class through serde, which has no
        /// way to pass the concrete Rust type along with the struct.
        pub(crate) fn complex_tag_for_sentinel(name: &str) -> Option<ComplexTag> {
            match name {
                $($sentinel => Some(ComplexTag::$tag),)*
                _ => None,
            }
        }
    };
}

complex_types! {
    Complex64 => f64, COMPLEX64_SENTINEL, F64, "double",
    Complex32 => f32, COMPLEX32_SENTINEL, F32, "single",
    ComplexI64 => i64, COMPLEX_I64_SENTINEL, I64, "int64",
    ComplexI32 => i32, COMPLEX_I32_SENTINEL, I32, "int32",
    ComplexI16 => i16, COMPLEX_I16_SENTINEL, I16, "int16",
    ComplexI8 => i8, COMPLEX_I8_SENTINEL, I8, "int8",
    ComplexU64 => u64, COMPLEX_U64_SENTINEL, U64, "uint64",
    ComplexU32 => u32, COMPLEX_U32_SENTINEL, U32, "uint32",
    ComplexU16 => u16, COMPLEX_U16_SENTINEL, U16, "uint16",
    ComplexU8 => u8, COMPLEX_U8_SENTINEL, U8, "uint8",
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_compare() {
        let a = Complex64::new(1.0, -2.0);
        let b = Complex64 { re: 1.0, im: -2.0 };
        assert_eq!(a, b);
    }

    #[test]
    fn each_sentinel_names_its_own_component_class() {
        assert_eq!(
            complex_tag_for_sentinel(COMPLEX_I16_SENTINEL),
            Some(ComplexTag::I16)
        );
        assert_eq!(
            complex_tag_for_sentinel(COMPLEX64_SENTINEL),
            Some(ComplexTag::F64)
        );
        assert_eq!(complex_tag_for_sentinel("SomeUserStruct"), None);
    }

    /// The sentinel is the only thing distinguishing one complex type from
    /// another once serde has erased the Rust type, so two sharing a name
    /// would silently write each other's class.
    #[test]
    fn sentinels_are_distinct() {
        let all = [
            COMPLEX64_SENTINEL,
            COMPLEX32_SENTINEL,
            COMPLEX_I64_SENTINEL,
            COMPLEX_I32_SENTINEL,
            COMPLEX_I16_SENTINEL,
            COMPLEX_I8_SENTINEL,
            COMPLEX_U64_SENTINEL,
            COMPLEX_U32_SENTINEL,
            COMPLEX_U16_SENTINEL,
            COMPLEX_U8_SENTINEL,
        ];
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(unique.len(), all.len());
    }
}
