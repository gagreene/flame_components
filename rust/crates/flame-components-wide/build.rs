fn main() {
    println!("cargo:rerun-if-changed=src/native.c");
    cc::Build::new()
        .file("src/native.c")
        .compile("flame_components_native_wide");
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
    }
}
