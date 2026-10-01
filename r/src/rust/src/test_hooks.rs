//! Test-only entry points, compiled with the `test-hooks` feature.
//!
//! `pixi run -e r r-install` builds with it (`PG_CARGO_FEATURES`); the source
//! tarball never does.  Without the feature the module registers nothing.

use extendr_api::prelude::*;

/// Panic inside a native call, so the package tests can show a Rust panic
/// reaches R as an error and leaves the session usable.
#[cfg(feature = "test-hooks")]
#[extendr]
fn panic_for_test() {
    panic!("pedigree-graph test hook: deliberate panic");
}

/// A pair total of `count` (a decimal string) as `relationship_counts`
/// hands it to R, so the tests can reach the 2^53 refusal.
#[cfg(feature = "test-hooks")]
#[extendr]
fn count_as_double_for_test(count: &str) -> Robj {
    let count: u64 = count.parse().expect("a decimal u64");
    crate::errors::finish(crate::kernels::exact_double("FS", count).map(Robj::from))
}

#[cfg(feature = "test-hooks")]
extendr_module! {
    mod test_hooks;
    fn panic_for_test;
    fn count_as_double_for_test;
}

#[cfg(not(feature = "test-hooks"))]
extendr_module! {
    mod test_hooks;
}
