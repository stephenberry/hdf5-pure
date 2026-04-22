//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::collections::HashMap;

use crate::attribute::extract_attributes_full;
use crate::chunk_cache::ChunkCache;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::filter_pipeline::FilterPipeline;
use crate::group_v1::GroupEntry;
use crate::group_v2;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::superblock::Superblock;

use crate::types::{attrs_to_map, classify_datatype, AttrValue, DType};

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// An open HDF5 file for reading.
pub struct File {
    data: Vec<u8>,
    superblock: Superblock,
    chunk_cache: ChunkCache,
    /// Byte offset to add to all relative addresses (= original base_address).
    addr_offset: u64,
}

impl File {
    /// Open an HDF5 file from a filesystem path.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        Self::from_bytes(bytes)
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        let sig_offset = signature::find_signature(&data)?;
        let mut superblock = Superblock::parse(&data, sig_offset)?;
        let addr_offset = superblock.base_address;
        // Normalize root_group_address to absolute so resolve_path_any works.
        superblock.root_group_address += addr_offset;
        Ok(Self {
            data,
            superblock,
            chunk_cache: ChunkCache::new(),
            addr_offset,
        })
    }

    /// Returns a handle to the root group.
    pub fn root(&self) -> Group<'_> {
        Group {
            file: self,
            // root_group_address was normalized to absolute in from_bytes()
            address: self.superblock.root_group_address,
        }
    }

    /// Resolve a path and return a `Dataset` handle.
    pub fn dataset(&self, path: &str) -> Result<Dataset<'_>, Error> {
        let addr = group_v2::resolve_path_any(&self.data, &self.superblock, path)?;
        let hdr = self.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(Dataset {
            file: self,
            header: hdr,
        })
    }

    /// Resolve a path and return a `Group` handle.
    pub fn group(&self, path: &str) -> Result<Group<'_>, Error> {
        let addr = group_v2::resolve_path_any(&self.data, &self.superblock, path)?;
        Ok(Group {
            file: self,
            address: addr,
        })
    }

    /// Returns the raw file bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        ObjectHeader::parse_with_base(
            &self.data,
            address as usize,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.addr_offset,
        )
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.data.len())
            .field("superblock_version", &self.superblock.version)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 group.
pub struct Group<'f> {
    file: &'f File,
    address: u64,
}

impl<'f> Group<'f> {
    /// List the names of datasets in this group.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// List the names of subgroups in this group.
    pub fn groups(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        let attr_msgs = extract_attributes_full(
            &self.file.data,
            &hdr,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            &self.file.data,
            self.file.offset_size(),
            self.file.length_size(),
            self.file.addr_offset,
        ))
    }

    /// Get a dataset within this group by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        Ok(Dataset {
            file: self.file,
            header: hdr,
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<Group<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(Group {
            file: self.file,
            address: entry.object_header_address,
        })
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        let os = self.file.offset_size();
        let ls = self.file.length_size();
        let base = self.file.addr_offset;
        let mut entries = group_v2::resolve_group_entries(&self.file.data, &hdr, os, ls, base)
            .map_err(Error::Format)?;
        // Convert link addresses from relative to absolute
        for entry in &mut entries {
            entry.object_header_address += base;
        }
        Ok(entries)
    }
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 dataset.
#[derive(Debug)]
pub struct Dataset<'f> {
    file: &'f File,
    header: ObjectHeader,
}

impl<'f> Dataset<'f> {
    /// Returns the shape (dimensions) of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        let ds = self.dataspace()?;
        Ok(ds.dimensions.clone())
    }

    /// Returns the simplified datatype of the dataset.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
    }

    /// Read all data as `f64` values.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64(&raw, &dt)?)
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32(&raw, &dt)?)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i32(&raw, &dt)?)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i64(&raw, &dt)?)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u64(&raw, &dt)?)
    }

    /// Read all data as `u8` values.
    pub fn read_u8(&self) -> Result<Vec<u8>, Error> {
        self.read_raw()
    }

    /// Read all data as `i8` values.
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw()?;
        Ok(raw.iter().map(|&b| b as i8).collect())
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_i32(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as i16).collect())
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_u64(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as u16).collect())
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_u64(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as u32).collect())
    }

    /// Read all data as `String` values.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_strings(&raw, &dt)?)
    }

    /// Read all attributes of this dataset.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let attr_msgs = extract_attributes_full(
            &self.file.data,
            &self.header,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            &self.file.data,
            self.file.offset_size(),
            self.file.length_size(),
            self.file.addr_offset,
        ))
    }

    fn datatype(&self) -> Result<Datatype, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&msg.data)?;
        Ok(dt)
    }

    fn dataspace(&self) -> Result<Dataspace, Error> {
        let msg = find_message(&self.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(&msg.data, self.file.length_size())?)
    }

    fn data_layout(&self) -> Result<DataLayout, Error> {
        let msg = find_message(&self.header, MessageType::DataLayout)?;
        Ok(DataLayout::parse(
            &msg.data,
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }

    fn filter_pipeline(&self) -> Option<FilterPipeline> {
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok())
    }

    fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let mut dl = self.data_layout()?;
        // Adjust contiguous data address by base_address offset
        if self.file.addr_offset != 0 {
            if let DataLayout::Contiguous {
                ref mut address, ..
            } = dl
            {
                if let Some(addr) = address {
                    *addr += self.file.addr_offset;
                }
            }
        }
        let pipeline = self.filter_pipeline();
        Ok(data_read::read_raw_data_cached(
            &self.file.data,
            &dl,
            &ds,
            &dt,
            pipeline.as_ref(),
            self.file.offset_size(),
            self.file.length_size(),
            &self.file.chunk_cache,
        )?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_message(
    header: &ObjectHeader,
    msg_type: MessageType,
) -> Result<&crate::object_header::HeaderMessage, Error> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == msg_type)
        .ok_or(Error::MissingMessage(msg_type))
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
}

fn is_group(header: &ObjectHeader) -> bool {
    header.messages.iter().any(|m| {
        m.msg_type == MessageType::LinkInfo
            || m.msg_type == MessageType::Link
            || m.msg_type == MessageType::SymbolTable
    })
}
