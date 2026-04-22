//! Build script.
//!
//! The only behaviour this script carries is emitting linker directives for
//! the **optional** `matio-crosscheck` test feature. When that feature is not
//! enabled (the default), the script exits immediately and the build is
//! identical to having no `build.rs` at all — the crate stays pure-Rust with
//! no C-side dependency.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Cargo sets `CARGO_FEATURE_<NAME>` (uppercased, hyphens to underscores)
    // for every enabled feature. Bail out when the crosscheck feature is off.
    if std::env::var_os("CARGO_FEATURE_MATIO_CROSSCHECK").is_none() {
        return;
    }

    configure_libmatio();
}

/// Emit `cargo:rustc-link-*` directives for the system libmatio.
///
/// Checks well-known Homebrew and MacPorts locations on macOS, otherwise
/// relies on the default linker search path (sufficient for Linux distro
/// packages that install `libmatio.so` under `/usr/lib/...`).
fn configure_libmatio() {
    let candidates = [
        "/opt/homebrew/opt/libmatio/lib", // macOS Apple Silicon Homebrew
        "/usr/local/opt/libmatio/lib",    // macOS Intel Homebrew
        "/opt/local/lib",                 // MacPorts
    ];
    for dir in candidates {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=matio");
}
