//! Mid-level builder API for MAT v7.3 files.
//!
//! [`MatBuilder`] wraps [`crate::FileBuilder`] and applies MATLAB conventions:
//! `MATLAB_class` attributes, `MATLAB_int_decode` flags, `MATLAB_empty`
//! markers, the lazy `#refs#` group for object references, and the
//! `#subsystem#/MCOS` opaque-class subsystem for `string` objects.
//!
//! Callers hand in MATLAB-shape `dims` (e.g. `[N, 1]` for a column vector);
//! the builder transposes to HDF5 storage shape internally. Numeric data is
//! taken by `&[T]` so callers don't need to allocate per dataset.
//!
//! Nested structs and cells use closures so the borrow checker can keep the
//! call tree straight:
//!
//! ```ignore
//! let mut mb = MatBuilder::new(Options::default());
//! mb.struct_("payload", |s| {
//!     s.write_scalar_f64("answer", 42.0)?;
//!     s.cell("entries", &[2, 1], |c| {
//!         c.push_scalar_f64(1.0)?;
//!         c.push_string("hi")?;
//!         Ok(())
//!     })?;
//!     Ok(())
//! })?;
//! let bytes = mb.finish()?;
//! ```
//!
//! The builder allocates exactly one `#refs#/ref_{id:016x}` per cell element
//! and one per string-object saveobj payload; both groups are created lazily
//! on first use. No intermediate value tree is allocated.

use std::collections::HashSet;

use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::dims::{STORAGE_DIMS_BUF_LEN, matrix_dims, storage_dims_u64_into, vector_dims};
use crate::mat::error::MatError;
use crate::mat::identifier::{dedupe_name, is_valid_name, sanitize_name};
use crate::mat::options::{Compression, EmptyMarkerEncoding, InvalidNamePolicy, Options};
use crate::mat::string_object;
use crate::mat::userblock::{self, USERBLOCK_SIZE};
use crate::mat::utf16;
use crate::type_builders::{DatasetBuilder, GroupBuilder};
use crate::writer::FileBuilder;

const REFS_GROUP: &str = "#refs#";
const SUBSYSTEM_GROUP: &str = "#subsystem#";

/// Top-level MAT v7.3 writer.
pub struct MatBuilder {
    file: FileBuilder,
    options: Options,
    /// Lazy `#refs#` group, created on first use.
    refs: Option<GroupBuilder>,
    /// Monotonic counter for `ref_{id:016x}` names.
    next_ref_id: u64,
    /// Saveobj payload paths registered by `write_string_object`. Each entry
    /// becomes one MCOS reference.
    string_object_payload_paths: Vec<String>,
    /// Names already used at the file root.
    root_used_names: HashSet<String>,
    /// Stack of open struct groups. The deepest is the current write target
    /// when [`next_target`] is `None`.
    open_structs: Vec<OpenStruct>,
    /// Single-use override for the next `write_*` / `struct_` / `cell` call.
    /// Set by [`CellWriter`] before pushing a cell element; consumed by the
    /// next write to redirect the dataset/group into `#refs#`.
    next_target: Option<NextTarget>,
}

struct OpenStruct {
    group: GroupBuilder,
    used_names: HashSet<String>,
    fields: Vec<String>,
    parent: ParentKind,
}

enum ParentKind {
    /// Closing this struct attaches the finished group to the file root.
    Root,
    /// Closing this struct attaches the finished group to the open-struct at
    /// the given index of `open_structs` (after this one is popped).
    StructAt(usize),
    /// Closing this struct attaches the finished group to the `#refs#` group
    /// (using the group's own name, which is `ref_NNNN`).
    Refs,
}

/// Single-use override consumed by the next write/open.
struct NextTarget {
    /// The ref name (e.g. `"ref_0000000000000000"`). The dataset/group will
    /// be placed in `#refs#` with this name.
    ref_name: String,
}

impl MatBuilder {
    /// Construct a new MAT writer.
    pub fn new(options: Options) -> Self {
        let mut file = FileBuilder::new();
        file.with_userblock(USERBLOCK_SIZE);
        Self {
            file,
            options,
            refs: None,
            next_ref_id: 0,
            string_object_payload_paths: Vec::new(),
            root_used_names: HashSet::new(),
            open_structs: Vec::new(),
            next_target: None,
        }
    }

    /// Borrow the configured options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Allocate a fresh `ref_{:016x}` name. Does not write anything.
    pub fn alloc_ref_name(&mut self) -> String {
        let name = format!("ref_{:016x}", self.next_ref_id);
        self.next_ref_id += 1;
        name
    }

    /// Get-or-create the `#refs#` group on the underlying file.
    fn refs_mut(&mut self) -> &mut GroupBuilder {
        if self.refs.is_none() {
            self.refs = Some(self.file.create_group(REFS_GROUP));
        }
        self.refs.as_mut().unwrap()
    }

    fn normalize_name(
        raw: &str,
        used: &mut HashSet<String>,
        policy: InvalidNamePolicy,
    ) -> Result<String, MatError> {
        let candidate = match policy {
            InvalidNamePolicy::Error => {
                if !is_valid_name(raw) {
                    return Err(MatError::Custom(format!("invalid MATLAB name: {raw}")));
                }
                raw.to_owned()
            }
            InvalidNamePolicy::Sanitize => sanitize_name(raw),
        };
        let unique = dedupe_name(candidate, used);
        used.insert(unique.clone());
        Ok(unique)
    }

    /// Resolve a name in the current write scope and return the (resolved
    /// name, write target) pair to use for the next dataset/group.
    fn resolve_target(&mut self, raw_name: &str) -> Result<TargetForName, MatError> {
        if let Some(t) = self.next_target.take() {
            // Cell-element / explicit-ref target. The user-supplied name is
            // ignored; the ref name is fixed.
            return Ok(TargetForName::Ref(t.ref_name));
        }
        if let Some(top) = self.open_structs.last_mut() {
            let name = Self::normalize_name(
                raw_name,
                &mut top.used_names,
                self.options.invalid_name_policy,
            )?;
            top.fields.push(name.clone());
            return Ok(TargetForName::Struct {
                name,
                index: self.open_structs.len() - 1,
            });
        }
        let name = Self::normalize_name(
            raw_name,
            &mut self.root_used_names,
            self.options.invalid_name_policy,
        )?;
        Ok(TargetForName::Root(name))
    }

    /// Mutably borrow a `DatasetBuilder` at the resolved target.
    fn dataset_at_target<'a>(&'a mut self, target: &'a TargetForName) -> &'a mut DatasetBuilder {
        match target {
            TargetForName::Root(name) => self.file.create_dataset(name),
            TargetForName::Struct { name, index } => {
                self.open_structs[*index].group.create_dataset(name)
            }
            TargetForName::Ref(ref_name) => self.refs_mut().create_dataset(ref_name),
        }
    }

    // -- public scalar writes ---------------------------------------------

    pub fn write_scalar_logical(&mut self, name: &str, value: bool) -> Result<&mut Self, MatError> {
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u8_data(&[u8::from(value)]).with_shape(&[1, 1]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Logical.as_str().into()),
        );
        ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
        Ok(self)
    }

    pub fn write_scalar_f64(&mut self, name: &str, value: f64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Double, |ds| {
            ds.with_f64_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_f32(&mut self, name: &str, value: f32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Single, |ds| {
            ds.with_f32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i8(&mut self, name: &str, value: i8) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int8, |ds| {
            ds.with_i8_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i16(&mut self, name: &str, value: i16) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int16, |ds| {
            ds.with_i16_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i32(&mut self, name: &str, value: i32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int32, |ds| {
            ds.with_i32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i64(&mut self, name: &str, value: i64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int64, |ds| {
            ds.with_i64_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u8(&mut self, name: &str, value: u8) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt8, |ds| {
            ds.with_u8_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u16(&mut self, name: &str, value: u16) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt16, |ds| {
            ds.with_u16_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u32(&mut self, name: &str, value: u32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt32, |ds| {
            ds.with_u32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u64(&mut self, name: &str, value: u64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt64, |ds| {
            ds.with_u64_data(&[value]).with_shape(&[1, 1]);
        })
    }

    fn write_scalar_inner<F>(
        &mut self,
        name: &str,
        class: MatClass,
        apply: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut DatasetBuilder),
    {
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        apply(ds);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        Ok(self)
    }

    // -- numeric arrays ----------------------------------------------------

    pub fn write_f64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[f64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Double,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_f64_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_f32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[f32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Single,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_f32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i8(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i8],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int8,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i8_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i16(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i16],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int16,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i16_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int32,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int64,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i64_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u8(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt8,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u8_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u16(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u16],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt16,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u16_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt32,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt64,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u64_data(data).with_shape(shape);
            },
        )
    }

    /// Write a logical array (MATLAB `logical`, stored as `uint8`).
    pub fn write_logical(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        if data.is_empty() {
            return self.write_empty_with_decode(name, MatClass::Logical, matlab_dims, Some(1));
        }
        self.write_array_inner(
            name,
            MatClass::Logical,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u8_data(data).with_shape(shape);
                ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
            },
        )
    }

    /// Write a `char` UTF-16 string (MATLAB `char`). Uses `[N, 1]` MATLAB
    /// shape (a row vector when read back, since HDF5 storage transposes).
    pub fn write_char(&mut self, name: &str, value: &str) -> Result<&mut Self, MatError> {
        let units = utf16::encode_utf16(value);
        if units.is_empty() {
            return self.write_empty_with_decode(name, MatClass::Char, &[0, 0], Some(2));
        }
        let n = units.len() as u64;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u16_data(&units).with_shape(&[n, 1]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Char.as_str().into()),
        );
        ds.set_attr("MATLAB_int_decode", AttrValue::I32(2));
        Ok(self)
    }

    /// Write a complex `f64` array.
    ///
    /// Empty inputs produce a zero-element compound dataset (no
    /// `MATLAB_empty` marker): MATLAB reads this back as an empty complex
    /// array of the right class. Non-empty inputs honor the configured
    /// compression settings.
    pub fn write_complex_f64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[(f64, f64)],
    ) -> Result<&mut Self, MatError> {
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let compression = self.options.compression;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_complex64_data(data).with_shape(storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Double.as_str().into()),
        );
        if !data.is_empty() {
            apply_deflate(ds, compression);
        }
        Ok(self)
    }

    /// Write a complex `f32` array. See [`write_complex_f64`] for empty-input
    /// and compression semantics.
    pub fn write_complex_f32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[(f32, f32)],
    ) -> Result<&mut Self, MatError> {
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let compression = self.options.compression;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_complex32_data(data).with_shape(storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Single.as_str().into()),
        );
        if !data.is_empty() {
            apply_deflate(ds, compression);
        }
        Ok(self)
    }

    /// Write a MATLAB `string` object. Allocates a `#refs#` payload entry and
    /// registers the object in the MCOS subsystem.
    pub fn write_string_object(
        &mut self,
        name: &str,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        // 1. Encode payload, write to #refs# under a fresh ref.
        let payload = string_object::encode_string_saveobj_payload(values, matlab_dims)?;
        let payload_ref = self.alloc_ref_name();
        let payload_path = format!("{REFS_GROUP}/{payload_ref}");
        let payload_shape: [u64; 2] = [1, payload.len() as u64];
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&payload_ref);
            ds.with_u64_data(&payload).with_shape(&payload_shape);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::UInt64.as_str().into()),
            );
        }

        // 2. Register the payload path; the resulting object id is 1-based.
        self.string_object_payload_paths.push(payload_path);
        let object_id = self.string_object_payload_paths.len() as u32;

        // 3. Emit the parent metadata dataset at the current scope.
        let metadata = string_object::create_string_object_metadata(object_id);
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u32_data(&metadata)
            .with_shape(&[1, metadata.len() as u64]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(string_object::MATLAB_CLASS_STRING.into()),
        );
        ds.set_attr(
            "MATLAB_object_decode",
            AttrValue::I32(string_object::MATLAB_OBJECT_DECODE_OPAQUE),
        );
        Ok(self)
    }

    /// Write an empty marker for the given class. `matlab_dims` is the
    /// MATLAB-shape (e.g. `[0, 0]`).
    pub fn write_empty(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        let int_decode = match class {
            MatClass::Logical => Some(1),
            MatClass::Char => Some(2),
            _ => None,
        };
        self.write_empty_with_decode(name, class, matlab_dims, int_decode)
    }

    /// Write a MATLAB `struct([])` empty marker.
    pub fn write_empty_struct_array(&mut self, name: &str) -> Result<&mut Self, MatError> {
        self.write_empty_with_decode(name, MatClass::Struct, &[0, 0], None)
    }

    fn write_empty_with_decode(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
        int_decode: Option<i32>,
    ) -> Result<&mut Self, MatError> {
        let target = self.resolve_target(name)?;
        let encoding = self.options.empty_marker_encoding;
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let ds = self.dataset_at_target(&target);
        match encoding {
            EmptyMarkerEncoding::ZeroElement => {
                let shape = storage_dims_u64_into(matlab_dims, &mut storage_buf);
                emit_zero_element(ds, class, shape);
            }
            EmptyMarkerEncoding::DataAsDims => {
                if matlab_dims.len() > STORAGE_DIMS_BUF_LEN {
                    let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
                    ds.with_u64_data(&dim_data)
                        .with_shape(&[dim_data.len() as u64]);
                } else {
                    let n = matlab_dims.len();
                    for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                        *slot = d as u64;
                    }
                    ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
                }
            }
        }
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        ds.set_attr("MATLAB_empty", AttrValue::U32(1));
        if let Some(d) = int_decode {
            ds.set_attr("MATLAB_int_decode", AttrValue::I32(d));
        }
        Ok(self)
    }

    fn write_array_inner<F>(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
        data_len: usize,
        apply: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut DatasetBuilder, &[u64]),
    {
        if data_len == 0 {
            return self.write_empty_with_decode(name, class, matlab_dims, None);
        }
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let compression = self.options.compression;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        apply(ds, storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        apply_deflate(ds, compression);
        Ok(self)
    }

    // -- struct nesting ----------------------------------------------------

    /// Open a struct group at the current scope. Use [`StructWriter`] inside
    /// the closure to write fields.
    pub fn struct_<F>(&mut self, name: &str, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        // Resolve target and parent kind.
        let (group_name, parent) = match self.next_target.take() {
            Some(t) => (t.ref_name.clone(), ParentKind::Refs),
            None => {
                if let Some(top) = self.open_structs.last_mut() {
                    let resolved = Self::normalize_name(
                        name,
                        &mut top.used_names,
                        self.options.invalid_name_policy,
                    )?;
                    top.fields.push(resolved.clone());
                    (resolved, ParentKind::StructAt(self.open_structs.len() - 1))
                } else {
                    let resolved = Self::normalize_name(
                        name,
                        &mut self.root_used_names,
                        self.options.invalid_name_policy,
                    )?;
                    (resolved, ParentKind::Root)
                }
            }
        };

        let group = GroupBuilder::new(&group_name);
        self.open_structs.push(OpenStruct {
            group,
            used_names: HashSet::new(),
            fields: Vec::new(),
            parent,
        });

        let res = {
            let mut sw = StructWriter { mb: self };
            fill(&mut sw)
        };

        // Always close even if fill errored, to keep the stack consistent.
        let close_res = self.close_struct();
        res?;
        close_res?;
        Ok(self)
    }

    fn close_struct(&mut self) -> Result<(), MatError> {
        let mut s = self
            .open_structs
            .pop()
            .ok_or_else(|| MatError::Custom("close_struct called with no open struct".into()))?;
        s.group.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Struct.as_str().into()),
        );
        s.group
            .set_attr("MATLAB_fields", AttrValue::VarLenAsciiArray(s.fields));
        let finished = s.group.finish();
        match s.parent {
            ParentKind::Root => self.file.add_group(finished),
            ParentKind::StructAt(idx) => self.open_structs[idx].group.add_group(finished),
            ParentKind::Refs => self.refs_mut().add_group(finished),
        }
        Ok(())
    }

    // -- cell arrays -------------------------------------------------------

    /// Write a cell array. The closure pushes elements; the parent dataset is
    /// emitted with object references after the closure returns.
    pub fn cell<F>(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        fill: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        // Resolve where the cell parent dataset goes BEFORE running fill (so
        // pushes from fill don't mutate the resolved scope).
        let parent_target = self.resolve_target(name)?;

        let mut paths: Vec<String> = Vec::new();
        let res = {
            let mut cw = CellWriter {
                mb: self,
                paths: &mut paths,
            };
            fill(&mut cw)
        };
        res?;

        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        if paths.is_empty() {
            // Empty cell: emit a class=cell empty marker at the resolved
            // target. Reuse the empty-marker emit path.
            let encoding = self.options.empty_marker_encoding;
            let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
            let ds = self.dataset_at_target(&parent_target);
            match encoding {
                EmptyMarkerEncoding::ZeroElement => {
                    emit_zero_element(ds, MatClass::Cell, storage);
                }
                EmptyMarkerEncoding::DataAsDims => {
                    if matlab_dims.len() > STORAGE_DIMS_BUF_LEN {
                        let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
                        ds.with_u64_data(&dim_data)
                            .with_shape(&[dim_data.len() as u64]);
                    } else {
                        let n = matlab_dims.len();
                        for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                            *slot = d as u64;
                        }
                        ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
                    }
                }
            }
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::Cell.as_str().into()),
            );
            ds.set_attr("MATLAB_empty", AttrValue::U32(1));
            return Ok(self);
        }

        let path_strs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let ds = self.dataset_at_target(&parent_target);
        ds.with_path_references(&path_strs).with_shape(storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Cell.as_str().into()),
        );
        Ok(self)
    }

    // -- helpers -----------------------------------------------------------

    /// MATLAB shape for a 1-D vector of length `len`, using the configured
    /// 1-D mode. Returns a stack-allocated 2-element array.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        vector_dims(len, self.options.one_dimensional_mode)
    }

    /// MATLAB shape for a value with the given multi-dimensional `extents`.
    /// Allocates only when `extents.len() > 2`; the 0/1/2-D cases construct
    /// a `Vec<usize>` with capacity 2.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        matrix_dims(extents, self.options.one_dimensional_mode)
    }

    // -- finalize ----------------------------------------------------------

    /// Finalize the file. Writes the `#subsystem#/MCOS` group if any string
    /// objects were written, the `#refs#` group if any refs were created,
    /// the userblock, and returns the bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, MatError> {
        if !self.open_structs.is_empty() {
            return Err(MatError::Custom(format!(
                "MatBuilder::finish called with {} open structs",
                self.open_structs.len()
            )));
        }
        if self.next_target.is_some() {
            return Err(MatError::Custom(
                "MatBuilder::finish called with a pending cell-element target".into(),
            ));
        }

        if !self.string_object_payload_paths.is_empty() {
            self.emit_subsystem()?;
        }

        if let Some(refs) = self.refs.take() {
            self.file.add_group(refs.finish());
        }

        let mut bytes = self.file.finish().map_err(MatError::Hdf5)?;
        userblock::write_header(&mut bytes, userblock::DEFAULT_DESCRIPTION);
        Ok(bytes)
    }

    /// Build the MCOS subsystem. Lifts the layout from the beve writer
    /// byte-for-byte: 5 helper refs (FileWrapper, canonical empty,
    /// unknown-template-A, alias-int32, unknown-template-B) plus the
    /// per-string saveobj payload paths.
    fn emit_subsystem(&mut self) -> Result<(), MatError> {
        // 1. FileWrapper__ metadata blob.
        let metadata = string_object::build_string_filewrapper_metadata(
            self.string_object_payload_paths.len(),
        );
        let metadata_ref = self.alloc_ref_name();
        let metadata_path = format!("{REFS_GROUP}/{metadata_ref}");
        let metadata_shape: [u64; 2] = [1, metadata.len() as u64];
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&metadata_ref);
            ds.with_u8_data(&metadata).with_shape(&metadata_shape);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::UInt8.as_str().into()),
            );
        }

        // 2. Canonical empty.
        let canonical_ref = self.alloc_ref_name();
        let canonical_path = format!("{REFS_GROUP}/{canonical_ref}");
        self.write_subsystem_empty_marker(&canonical_ref, &[0, 0], "canonical empty", None);

        // 3. Unknown template A: cell with two empty struct refs.
        let empty_a1 = self.alloc_ref_name();
        let empty_a1_path = format!("{REFS_GROUP}/{empty_a1}");
        self.write_subsystem_empty_marker(&empty_a1, &[1, 0], "struct", None);

        let empty_a2 = self.alloc_ref_name();
        let empty_a2_path = format!("{REFS_GROUP}/{empty_a2}");
        self.write_subsystem_empty_marker(&empty_a2, &[1, 0], "struct", None);

        let template_a = self.alloc_ref_name();
        let template_a_path = format!("{REFS_GROUP}/{template_a}");
        self.write_subsystem_reference_array(
            &template_a,
            &[2, 1],
            &[empty_a1_path.as_str(), empty_a2_path.as_str()],
            "cell",
        );

        // 4. Alias int32 ref.
        let alias_ref = self.alloc_ref_name();
        let alias_path = format!("{REFS_GROUP}/{alias_ref}");
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&alias_ref);
            ds.with_i32_data(&[0i32, 0]).with_shape(&[1u64, 2]);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::Int32.as_str().into()),
            );
        }

        // 5. Unknown template B: cell with two empty struct refs.
        let empty_b1 = self.alloc_ref_name();
        let empty_b1_path = format!("{REFS_GROUP}/{empty_b1}");
        self.write_subsystem_empty_marker(&empty_b1, &[1, 0], "struct", None);

        let empty_b2 = self.alloc_ref_name();
        let empty_b2_path = format!("{REFS_GROUP}/{empty_b2}");
        self.write_subsystem_empty_marker(&empty_b2, &[1, 0], "struct", None);

        let template_b = self.alloc_ref_name();
        let template_b_path = format!("{REFS_GROUP}/{template_b}");
        self.write_subsystem_reference_array(
            &template_b,
            &[2, 1],
            &[empty_b1_path.as_str(), empty_b2_path.as_str()],
            "cell",
        );

        // 6. Build the MCOS reference array.
        let mut paths: Vec<String> = Vec::with_capacity(5 + self.string_object_payload_paths.len());
        paths.push(metadata_path);
        paths.push(canonical_path);
        paths.extend(self.string_object_payload_paths.iter().cloned());
        paths.push(template_a_path);
        paths.push(alias_path);
        paths.push(template_b_path);

        let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let mut subsystem_group = self.file.create_group(SUBSYSTEM_GROUP);
        let ds = subsystem_group.create_dataset("MCOS");
        ds.with_path_references(&path_refs)
            .with_shape(&[1u64, paths.len() as u64]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(string_object::MATLAB_CLASS_FILEWRAPPER.into()),
        );
        ds.set_attr(
            "MATLAB_object_decode",
            AttrValue::I32(string_object::MATLAB_OBJECT_DECODE_OPAQUE),
        );
        self.file.add_group(subsystem_group.finish());
        Ok(())
    }

    /// Subsystem helper refs always use the data-as-dims encoding regardless
    /// of `Options::empty_marker_encoding`: real MATLAB writes them this way
    /// in the `mxOPAQUE_CLASS` machinery, and we lift the layout byte-for-byte.
    fn write_subsystem_empty_marker(
        &mut self,
        ref_name: &str,
        matlab_dims: &[usize],
        class: &str,
        int_decode: Option<i32>,
    ) {
        let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let n = matlab_dims.len();
        let refs = self.refs_mut();
        let ds = refs.create_dataset(ref_name);
        if n > STORAGE_DIMS_BUF_LEN {
            let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
            ds.with_u64_data(&dim_data).with_shape(&[n as u64]);
        } else {
            for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                *slot = d as u64;
            }
            ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
        }
        ds.set_attr("MATLAB_class", AttrValue::AsciiString(class.into()));
        ds.set_attr("MATLAB_empty", AttrValue::U32(1));
        if let Some(d) = int_decode {
            ds.set_attr("MATLAB_int_decode", AttrValue::I32(d));
        }
    }

    fn write_subsystem_reference_array(
        &mut self,
        ref_name: &str,
        matlab_dims: &[usize],
        paths: &[&str],
        class: &str,
    ) {
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let refs = self.refs_mut();
        let ds = refs.create_dataset(ref_name);
        ds.with_path_references(paths).with_shape(storage);
        ds.set_attr("MATLAB_class", AttrValue::AsciiString(class.into()));
    }
}

/// Where to put the next dataset/group, as resolved from current scope and
/// `next_target`.
enum TargetForName {
    Root(String),
    Struct { name: String, index: usize },
    Ref(String),
}

/// Apply the configured deflate compression to `ds`, if any.
fn apply_deflate(ds: &mut DatasetBuilder, compression: Compression) {
    if let Compression::Deflate { level, shuffle } = compression {
        if shuffle {
            ds.with_shuffle();
        }
        ds.with_deflate(level as u32);
    }
}

fn emit_zero_element(ds: &mut DatasetBuilder, class: MatClass, shape: &[u64]) {
    match class {
        MatClass::Double => {
            ds.with_f64_data(&[]).with_shape(shape);
        }
        MatClass::Single => {
            ds.with_f32_data(&[]).with_shape(shape);
        }
        MatClass::Int8 => {
            ds.with_i8_data(&[]).with_shape(shape);
        }
        MatClass::Int16 => {
            ds.with_i16_data(&[]).with_shape(shape);
        }
        MatClass::Int32 => {
            ds.with_i32_data(&[]).with_shape(shape);
        }
        MatClass::Int64 => {
            ds.with_i64_data(&[]).with_shape(shape);
        }
        MatClass::UInt8 | MatClass::Logical | MatClass::Cell | MatClass::Struct => {
            ds.with_u8_data(&[]).with_shape(shape);
        }
        MatClass::UInt16 | MatClass::Char => {
            ds.with_u16_data(&[]).with_shape(shape);
        }
        MatClass::UInt32 => {
            ds.with_u32_data(&[]).with_shape(shape);
        }
        MatClass::UInt64 => {
            ds.with_u64_data(&[]).with_shape(shape);
        }
    }
}

/// Scoped writer inside a struct group. Mirrors [`MatBuilder`]'s public write
/// methods but writes to the parent struct.
pub struct StructWriter<'a> {
    mb: &'a mut MatBuilder,
}

impl<'a> StructWriter<'a> {
    /// Borrow the underlying [`MatBuilder`]. Use this when a generic walker
    /// needs to write into the current scope without going through the
    /// struct-specific forwarding methods. The builder's internal scope
    /// stack ensures writes target the open struct group.
    #[inline]
    pub fn builder(&mut self) -> &mut MatBuilder {
        self.mb
    }

    #[inline]
    pub fn write_scalar_logical(&mut self, name: &str, value: bool) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_logical(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_f64(&mut self, name: &str, value: f64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_f64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_f32(&mut self, name: &str, value: f32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_f32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i8(&mut self, name: &str, value: i8) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i8(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i16(&mut self, name: &str, value: i16) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i16(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i32(&mut self, name: &str, value: i32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i64(&mut self, name: &str, value: i64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u8(&mut self, name: &str, value: u8) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u8(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u16(&mut self, name: &str, value: u16) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u16(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u32(&mut self, name: &str, value: u32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u64(&mut self, name: &str, value: u64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_f64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[f64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_f64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_f32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[f32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_f32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i8(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i8(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i16(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i16],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i16(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u8(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u8(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u16(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u16],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u16(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_logical(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_logical(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_char(&mut self, name: &str, value: &str) -> Result<&mut Self, MatError> {
        self.mb.write_char(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_complex_f64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[(f64, f64)],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_complex_f64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_complex_f32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[(f32, f32)],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_complex_f32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_string_object(
        &mut self,
        name: &str,
        values: &[String],
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_string_object(name, values, dims)?;
        Ok(self)
    }
    #[inline]
    pub fn write_empty(
        &mut self,
        name: &str,
        class: MatClass,
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_empty(name, class, dims)?;
        Ok(self)
    }
    #[inline]
    pub fn write_empty_struct_array(&mut self, name: &str) -> Result<&mut Self, MatError> {
        self.mb.write_empty_struct_array(name)?;
        Ok(self)
    }
    pub fn struct_<F>(&mut self, name: &str, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        self.mb.struct_(name, fill)?;
        Ok(self)
    }
    pub fn cell<F>(&mut self, name: &str, dims: &[usize], fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        self.mb.cell(name, dims, fill)?;
        Ok(self)
    }

    /// MATLAB shape for a 1-D vector.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        self.mb.vector_dims(len)
    }
    /// MATLAB shape for multi-dimensional extents.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        self.mb.matrix_dims(extents)
    }
    /// Access the underlying options.
    #[inline]
    pub fn options(&self) -> &Options {
        self.mb.options()
    }
    /// Convenience: configured [`StringClass`].
    #[inline]
    pub fn string_class(&self) -> crate::mat::options::StringClass {
        self.options().string_class
    }
}

/// Scoped writer inside a cell array. Each `push_*` allocates a fresh
/// `#refs#/ref_NNNN` and records its absolute path. The cell's parent dataset
/// is emitted when the closure returns.
pub struct CellWriter<'a> {
    mb: &'a mut MatBuilder,
    paths: &'a mut Vec<String>,
}

impl<'a> CellWriter<'a> {
    fn arm(&mut self) -> String {
        let ref_name = self.mb.alloc_ref_name();
        self.mb.next_target = Some(NextTarget {
            ref_name: ref_name.clone(),
        });
        ref_name
    }

    fn record(&mut self, ref_name: String) {
        // Defensive: if the write didn't consume next_target (shouldn't
        // happen), drop it so the next push starts clean.
        let _ = self.mb.next_target.take();
        self.paths.push(format!("{REFS_GROUP}/{ref_name}"));
    }

    pub fn push_scalar_logical(&mut self, value: bool) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_logical(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_f64(&mut self, value: f64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_f64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_f32(&mut self, value: f32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_f32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u8(&mut self, value: u8) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u8(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u16(&mut self, value: u16) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u16(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u32(&mut self, value: u32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u64(&mut self, value: u64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i8(&mut self, value: i8) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i8(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i16(&mut self, value: i16) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i16(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i32(&mut self, value: i32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i64(&mut self, value: i64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_string(&mut self, value: &str) -> Result<&mut Self, MatError> {
        // MATLAB string scalar: dims [1, 1].
        let r = self.arm();
        self.mb
            .write_string_object(&r, &[value.to_owned()], &[1, 1])?;
        self.record(r);
        Ok(self)
    }
    pub fn push_char(&mut self, value: &str) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_char(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_f64(&mut self, dims: &[usize], data: &[f64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_f64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_f32(&mut self, dims: &[usize], data: &[f32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_f32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i8(&mut self, dims: &[usize], data: &[i8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i8(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i16(&mut self, dims: &[usize], data: &[i16]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i16(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i32(&mut self, dims: &[usize], data: &[i32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i64(&mut self, dims: &[usize], data: &[i64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u8(&mut self, dims: &[usize], data: &[u8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u8(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u16(&mut self, dims: &[usize], data: &[u16]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u16(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u32(&mut self, dims: &[usize], data: &[u32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u64(&mut self, dims: &[usize], data: &[u64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_logical(&mut self, dims: &[usize], data: &[u8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_logical(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_complex_f64(
        &mut self,
        dims: &[usize],
        data: &[(f64, f64)],
    ) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_complex_f64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_complex_f32(
        &mut self,
        dims: &[usize],
        data: &[(f32, f32)],
    ) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_complex_f32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_string_object(
        &mut self,
        values: &[String],
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_string_object(&r, values, dims)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_empty(&mut self, class: MatClass, dims: &[usize]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_empty(&r, class, dims)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_empty_struct_array(&mut self) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_empty_struct_array(&r)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_struct<F>(&mut self, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        let r = self.arm();
        self.mb.struct_(&r, fill)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_cell<F>(&mut self, dims: &[usize], fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        let r = self.arm();
        self.mb.cell(&r, dims, fill)?;
        self.record(r);
        Ok(self)
    }

    /// Push an explicit reference to an existing path (e.g. a previously
    /// allocated `#refs#/ref_NNNN` written via [`MatBuilder::alloc_ref_name`]
    /// and out-of-band logic). For advanced use only.
    pub fn push_path(&mut self, path: String) -> &mut Self {
        self.paths.push(path);
        self
    }

    /// Allocate a fresh ref and run `build` with the underlying `MatBuilder`,
    /// armed so the next `write_*` / `struct_` / `cell` call inside `build`
    /// targets `#refs#/ref_NNNN`. The build closure is responsible for
    /// performing exactly one write/struct/cell operation.
    pub fn push_with<F>(&mut self, build: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut MatBuilder) -> Result<(), MatError>,
    {
        let r = self.arm();
        let res = build(self.mb);
        // `record` clears any leftover next_target as a safety net.
        self.record(r);
        res?;
        Ok(self)
    }

    /// Access the configured options.
    #[inline]
    pub fn options(&self) -> &Options {
        self.mb.options()
    }

    /// MATLAB shape for a 1-D vector under the configured mode.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        self.mb.vector_dims(len)
    }

    /// MATLAB shape for multi-dimensional extents under the configured mode.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        self.mb.matrix_dims(extents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::File;
    use crate::types::AttrValue as ReaderAttr;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("hdf5pure-mat-{name}-{nanos}.mat"))
    }

    fn read_class(file: &File, ds_path: &str) -> String {
        let ds = file.dataset(ds_path).unwrap();
        let attrs = ds.attrs().unwrap();
        match &attrs["MATLAB_class"] {
            ReaderAttr::AsciiString(s) | ReaderAttr::String(s) => s.clone(),
            other => panic!("unexpected class: {other:?}"),
        }
    }

    #[test]
    fn scalar_f64_at_root() {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_scalar_f64("x", 1.5).unwrap();
        let bytes = mb.finish().unwrap();

        let path = temp_path("scalar-f64");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let ds = file.dataset("x").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![1.5]);
        assert_eq!(read_class(&file, "x"), "double");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn nested_struct_writes_fields() {
        let mut mb = MatBuilder::new(Options::default());
        mb.struct_("payload", |s| {
            s.write_scalar_u32("answer", 7)?;
            s.write_char("label", "hello")?;
            Ok(())
        })
        .unwrap();
        let bytes = mb.finish().unwrap();

        let path = temp_path("struct");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let group = file.group("payload").unwrap();
        let attrs = group.attrs().unwrap();
        let class = match &attrs["MATLAB_class"] {
            ReaderAttr::AsciiString(s) | ReaderAttr::String(s) => s.clone(),
            other => panic!("unexpected: {other:?}"),
        };
        assert_eq!(class, "struct");
        assert_eq!(read_class(&file, "payload/answer"), "uint32");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cell_with_two_refs() {
        let mut mb = MatBuilder::new(Options::default());
        mb.cell("c", &[2, 1], |cw| {
            cw.push_scalar_u8(1)?;
            cw.push_scalar_u8(2)?;
            Ok(())
        })
        .unwrap();
        let bytes = mb.finish().unwrap();

        let path = temp_path("cell");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let cls = read_class(&file, "c");
        assert_eq!(cls, "cell");
        assert_eq!(read_class(&file, "#refs#/ref_0000000000000000"), "uint8");
        assert_eq!(read_class(&file, "#refs#/ref_0000000000000001"), "uint8");
        std::fs::remove_file(path).unwrap();
    }
}
