#[cfg(feature = "extended")]
fn main() {
    println!("cargo:rerun-if-changed=src/wide.c");
    cc::Build::new()
        .file("src/wide.c")
        .compile("flame_components_native_wide");
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
    }
}

#[cfg(not(feature = "extended"))]
fn main() {}
