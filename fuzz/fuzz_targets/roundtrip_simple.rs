#![no_main]

use arbitrary::Arbitrary;
use hdf5_pure::{DType, File, FileBuilder};
use libfuzzer_sys::fuzz_target;

const MAX_ELEMENTS: usize = 32;

#[derive(Arbitrary, Debug)]
struct Case {
    in_group: bool,
    dataset: DatasetCase,
}

#[derive(Arbitrary, Debug)]
enum DatasetCase {
    I32(Vec<i32>),
    U8(Vec<u8>),
    F64(Vec<u64>),
}

fuzz_target!(|case: Case| {
    let expected = ExpectedDataset::from(case.dataset);
    let bytes = build_file(case.in_group, &expected);
    let Ok(file) = File::from_bytes(bytes) else {
        panic!("writer produced unreadable file");
    };

    let path = if case.in_group { "group/data" } else { "data" };
    let dataset = file.dataset(path).expect("written dataset is missing");
    assert_eq!(dataset.shape().expect("shape"), vec![expected.len()]);
    assert_eq!(dataset.dtype().expect("dtype"), expected.dtype());
    expected.assert_readback(&dataset);
});

enum ExpectedDataset {
    I32(Vec<i32>),
    U8(Vec<u8>),
    F64(Vec<f64>),
}

impl ExpectedDataset {
    fn from(case: DatasetCase) -> Self {
        match case {
            DatasetCase::I32(mut values) => {
                values.truncate(MAX_ELEMENTS);
                Self::I32(values)
            }
            DatasetCase::U8(mut values) => {
                values.truncate(MAX_ELEMENTS);
                Self::U8(values)
            }
            DatasetCase::F64(values) => Self::F64(
                values
                    .into_iter()
                    .take(MAX_ELEMENTS)
                    .map(f64::from_bits)
                    .collect(),
            ),
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::I32(values) => values.len() as u64,
            Self::U8(values) => values.len() as u64,
            Self::F64(values) => values.len() as u64,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::I32(_) => DType::I32,
            Self::U8(_) => DType::U8,
            Self::F64(_) => DType::F64,
        }
    }

    fn assert_readback(&self, dataset: &hdf5_pure::Dataset<'_>) {
        match self {
            Self::I32(expected) => {
                assert_eq!(dataset.read_i32().expect("read i32"), *expected);
            }
            Self::U8(expected) => {
                assert_eq!(dataset.read_u8().expect("read u8"), *expected);
            }
            Self::F64(expected) => {
                let actual = dataset.read_f64().expect("read f64");
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }
}

fn build_file(in_group: bool, expected: &ExpectedDataset) -> Vec<u8> {
    macro_rules! write_dataset {
        ($builder:expr) => {
            match expected {
                ExpectedDataset::I32(values) => {
                    $builder.with_i32_data(values);
                }
                ExpectedDataset::U8(values) => {
                    $builder.with_u8_data(values);
                }
                ExpectedDataset::F64(values) => {
                    $builder.with_f64_data(values);
                }
            }
        };
    }

    let mut builder = FileBuilder::new();
    if in_group {
        let mut group = builder.create_group("group");
        write_dataset!(group.create_dataset("data"));
        builder.add_group(group.finish());
    } else {
        write_dataset!(builder.create_dataset("data"));
    }
    builder.finish().expect("writer failed")
}
