#![no_main]

use hdf5_pure::File;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = File::from_bytes(data.to_vec()) {
        let _ = file.superblock();
        let _ = file.root().datasets();
        let _ = file.root().groups();
        let _ = file.root().attrs();
    }
});
