#![no_main]

use hdf5_pure::{DType, Dataset, File, Group};
use libfuzzer_sys::fuzz_target;

const MAX_GROUP_DEPTH: usize = 3;
const MAX_CHILDREN_PER_GROUP: usize = 16;
const MAX_READ_ELEMENTS: u64 = 64;
/// Upper bound on the bytes a single exercised read may materialize. The element
/// count alone does not bound the allocation: a crafted datatype can declare a
/// per-element size of billions of bytes (a fixed-length string element, say), so
/// a few elements still multiply into a multi-gigabyte buffer and an out-of-memory
/// abort (issue #185). Gating on `element_count * element_size` keeps the target
/// exercising the decode paths on small data instead of re-discovering allocation
/// limits.
const MAX_READ_BYTES: u64 = 1 << 20; // 1 MiB

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = File::from_bytes(data.to_vec()) {
        let _ = file.superblock();
        exercise_group(&file, &file.root(), "", 0);
    }
});

fn exercise_group(file: &File, group: &Group, path: &str, depth: usize) {
    let _ = group.attrs();

    if let Ok(datasets) = group.datasets() {
        for name in datasets.iter().take(MAX_CHILDREN_PER_GROUP) {
            if let Ok(dataset) = group.dataset(name) {
                exercise_dataset(&dataset);
            }

            let full_path = child_path(path, name);
            if let Ok(dataset) = file.dataset(&full_path) {
                exercise_dataset(&dataset);
            }
        }
    }

    if depth >= MAX_GROUP_DEPTH {
        return;
    }

    if let Ok(groups) = group.groups() {
        for name in groups.iter().take(MAX_CHILDREN_PER_GROUP) {
            if let Ok(child) = group.group(name) {
                let full_path = child_path(path, name);
                exercise_group(file, &child, &full_path, depth + 1);
            }
        }
    }
}

fn exercise_dataset(dataset: &Dataset) {
    let shape = dataset.shape();
    let dtype = dataset.dtype();
    let _ = dataset.attrs();
    let _ = dataset.chunk_cache_stats();

    let Ok(shape) = shape else {
        return;
    };
    let Ok(dtype) = dtype else {
        return;
    };

    let element_count = shape
        .iter()
        .copied()
        .try_fold(1u64, |acc, dim| acc.checked_mul(dim));
    let Some(element_count @ 0..=MAX_READ_ELEMENTS) = element_count else {
        return;
    };

    // Bound the read by its byte size, not just its element count: the datatype's
    // on-disk element size is attacker-controlled, so a tiny element count can
    // still declare a huge allocation.
    let Ok(element_size) = dataset.element_size() else {
        return;
    };
    if element_count
        .checked_mul(element_size)
        .is_none_or(|bytes| bytes > MAX_READ_BYTES)
    {
        return;
    }

    match dtype {
        DType::F32 => {
            let _ = dataset.read_f32();
        }
        DType::F64 => {
            let _ = dataset.read_f64();
        }
        DType::I8 => {
            let _ = dataset.read_i8();
        }
        DType::I16 => {
            let _ = dataset.read_i16();
        }
        DType::I32 => {
            let _ = dataset.read_i32();
        }
        DType::I64 => {
            let _ = dataset.read_i64();
        }
        DType::U8 => {
            let _ = dataset.read_u8();
        }
        DType::U16 => {
            let _ = dataset.read_u16();
        }
        DType::U32 => {
            let _ = dataset.read_u32();
        }
        DType::U64 => {
            let _ = dataset.read_u64();
        }
        DType::String => {
            let _ = dataset.read_string();
        }
        _ => {}
    }
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    }
}
