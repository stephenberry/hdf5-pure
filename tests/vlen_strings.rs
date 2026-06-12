use hdf5_pure::{Error, File, FormatError, VlenStringReadOptions};

const FIXTURE: &str = "tests/fixtures/vl_strings.h5";

fn expected_names() -> Vec<String> {
    ["Alice", "Bob", "Charlie"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn buffered_vlen_dataset_read_and_size() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();

    assert_eq!(dataset.vlen_string_payload_size().unwrap(), 15);
    assert_eq!(dataset.read_string().unwrap(), expected_names());
    assert_eq!(
        dataset
            .read_vlen_strings(
                VlenStringReadOptions::new()
                    .with_max_elements(3)
                    .with_max_payload_bytes(15),
            )
            .unwrap(),
        expected_names()
    );
}

#[test]
fn vlen_element_limit_is_checked_before_payload_read() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_elements(2))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthElementLimitExceeded {
            limit: 2,
            actual: 3,
        })
    ));
}

#[test]
fn vlen_payload_limit_reports_required_bytes() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_payload_bytes(14))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthByteLimitExceeded {
            limit: 14,
            required: 15,
        })
    ));
}

#[test]
fn vlen_visitor_delivers_strings_in_order() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let mut names = Vec::new();

    dataset
        .visit_vlen_strings(
            VlenStringReadOptions::new().with_max_payload_bytes(15),
            |name| names.push(name.to_owned()),
        )
        .unwrap();

    assert_eq!(names, expected_names());
}

#[test]
fn vlen_specific_apis_reject_non_vlen_datasets() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let datatype = dataset.datatype().unwrap();
    assert!(matches!(
        datatype,
        hdf5_pure::Datatype::VariableLength { .. }
    ));

    let numeric = File::open("tests/fixtures/simple_dataset.h5").unwrap();
    let numeric_dataset = numeric.dataset("data").unwrap();
    assert!(matches!(
        numeric_dataset.vlen_string_payload_size(),
        Err(Error::Format(FormatError::TypeMismatch { .. }))
    ));
}
