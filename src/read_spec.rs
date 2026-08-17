//! The per-dataset properties every raw read is performed against.
//!
//! Reading a dataset's element bytes takes five facts about the dataset — where
//! its storage is, what shape it has, what an element is, what filters its bytes
//! passed through, and what unallocated storage reads as — and the call chain
//! that does it is four layers deep: the reader picks a backend, `data_read`
//! dispatches on the layout, `chunked_read` walks the chunk index, and the
//! assembly kernel places what it decoded. Every layer needs all five, so before
//! this type they were threaded through as five positional parameters and the
//! signatures reached twelve arguments.
//!
//! The cost of that shape was not the width; it was that adding a property meant
//! editing roughly thirty call sites, several of them mechanically across test
//! fixtures, where a default inserted in place of a real value compiles and
//! passes (issue #294). A property added to [`RawReadSpec`] instead reaches every
//! layer at once, and the sites that must decide what it is are the few that
//! build the spec.

use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;

/// What a raw read needs to know about the dataset it is reading.
///
/// These are properties of the *dataset*. The file-level address widths
/// (`offset_size`, `length_size`) travel beside a spec rather than in it: they
/// are the same for every dataset in a file, and the rest of the crate already
/// passes them as a pair.
///
/// `Copy`, so a spec is handed down the call chain the way the five borrows it
/// replaced were, with no clone and no lifetime beyond the borrows themselves.
#[derive(Clone, Copy)]
pub struct RawReadSpec<'a> {
    /// Where the dataset's bytes live: compact, contiguous, or chunked.
    pub layout: &'a DataLayout,
    /// The dataset's shape, which fixes how many elements the read produces.
    pub dataspace: &'a Dataspace,
    /// The stored element type, which fixes each element's width.
    pub datatype: &'a Datatype,
    /// The filter pipeline stored bytes passed through, if any.
    pub pipeline: Option<&'a FilterPipeline>,
    /// What storage that was never allocated reads as.
    pub fill: FillPattern<'a>,
}

impl<'a> RawReadSpec<'a> {
    /// A spec for an unfiltered dataset whose unallocated storage reads as
    /// zeros — the shape a test fixture wants, and never the shape a real read
    /// wants, which is why the live paths build the struct literally.
    #[cfg(test)]
    pub fn plain(layout: &'a DataLayout, dataspace: &'a Dataspace, datatype: &'a Datatype) -> Self {
        Self {
            layout,
            dataspace,
            datatype,
            pipeline: None,
            fill: FillPattern::ZERO,
        }
    }
}
