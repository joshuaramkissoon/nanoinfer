//! Bakes the toolchain and build flags into the binary, so every bench record
//! can say exactly how the code that produced it was compiled.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default().replace('\x1f', " ");
    println!("cargo:rustc-env=NANOINFER_RUSTC={version}");
    println!("cargo:rustc-env=NANOINFER_PROFILE={}", std::env::var("PROFILE").unwrap_or_default());
    println!("cargo:rustc-env=NANOINFER_RUSTFLAGS={flags}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
}
