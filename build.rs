fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let obj = format!("{out}/ldsum.o");
    let status = std::process::Command::new("cc")
        .args(["-c", "-O2", "src/ldsum.c", "-o", &obj])
        .status()
        .expect("cc is required to match R long-double cumsum");
    assert!(status.success(), "failed to compile src/ldsum.c");
    println!("cargo:rustc-link-arg={obj}");
    println!("cargo:rerun-if-changed=src/ldsum.c");
}
