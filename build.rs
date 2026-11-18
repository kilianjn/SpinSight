//! Build script: gives the Windows .exe its icon and version information.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rerun-if-changed=assets/icon/spinsight.ico");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon/spinsight.ico");
        res.set("ProductName", "SpinSight");
        res.set("FileDescription", "SpinSight – MRI simulator");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the Windows icon: {e}");
        }
    }
}
