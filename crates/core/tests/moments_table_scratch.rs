//! The heap a moments-table operation holds beyond its input and output
//! (ADR 0015): every accumulator is decoded when it is read, so the scratch
//! is a few big integers whatever the table's size.  A counting allocator
//! is this binary's global allocator, so the test sees every byte core
//! asks for.

use pedigree_graph_core::relationships::{MomentsTable, Product, Side, Statistic};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(live, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted as a fresh block alongside the old one: the bound holds
        // even if the allocator copies.
        grow(new_size);
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        out
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes allocated at the peak of `f` beyond what was live when it started.
fn peak_growth<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let start = LIVE.load(Ordering::SeqCst);
    PEAK.store(start, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - start)
}

const SCRATCH: usize = 64 << 10;

const PRODUCTS: [Product; 2] = [
    Product {
        a: (Side::First, 0),
        b: (Side::Second, 0),
    },
    Product {
        a: (Side::First, 1),
        b: (Side::Second, 1),
    },
];

/// A `[7, cells]` table of two columns and two products, accumulators of
/// up to 90 bits, as the engine hands it over.
fn table(cells: usize, exponents: [i64; 2]) -> MomentsTable<'static> {
    let stride = 1 + 4 * 2 + PRODUCTS.len();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let values: Vec<i128> = (0..7 * cells * stride)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if i % stride == 0 {
                i128::from(state >> 40)
            } else {
                (i128::from(state) << 26) - (1i128 << 89)
            }
        })
        .collect();
    MomentsTable::from_i128(
        vec![7, cells],
        2,
        PRODUCTS.to_vec(),
        exponents.to_vec(),
        &values,
    )
    .unwrap()
}

fn check_derivations(t: &MomentsTable<'_>, cells: usize) {
    for statistic in Statistic::ALL {
        let (out, growth) = peak_growth(|| t.derive(statistic, 1, "x").unwrap());
        let output = out.capacity() * 8;
        assert!(
            growth <= output + SCRATCH,
            "{} at {cells} cells, width {}: {growth} bytes for a {output}-byte output",
            statistic.name(),
            t.width()
        );
    }
}

fn check(cells: usize) {
    let t = table(cells, [40, -3]);
    let accumulators = t.bytes().len() / t.width();
    assert_eq!(t.width(), 12);
    check_derivations(&t, cells);

    // A sum over 7 categories needs at most one more byte per accumulator
    // before the output is narrowed.
    let (folded, growth) = peak_growth(|| t.sum(0).unwrap());
    let upper = accumulators / 7 * (t.width() + 1);
    assert!(
        growth <= upper + SCRATCH,
        "sum at {cells} cells: {growth} bytes, output bound {upper}"
    );
    assert_eq!(folded.shape(), &[cells]);

    // Equal exponents: the merge grows by its one carry byte.
    let (merged, growth) = peak_growth(|| t.merge(&t).unwrap());
    let upper = accumulators * (t.width() + 1);
    assert!(
        growth <= upper + SCRATCH,
        "merge at {cells} cells: {growth} bytes, output bound {upper}"
    );
    assert_eq!(merged.counts()[0], 2 * t.counts()[0]);

    let (_, growth) = peak_growth(|| t.total_pairs());
    assert!(
        growth <= SCRATCH,
        "total_pairs at {cells} cells: {growth} bytes"
    );

    // Exponents 300 bits apart: past 16 bytes, so every read is a big integer.
    let wide = t.merge(&table(cells, [340, -3])).unwrap();
    assert!(wide.width() > 16);
    check_derivations(&wide, cells);
    let (_, growth) = peak_growth(|| wide.sum(0).unwrap());
    let upper = accumulators / 7 * (wide.width() + 1);
    assert!(
        growth <= upper + SCRATCH,
        "wide sum at {cells} cells: {growth} bytes, output bound {upper}"
    );
}

#[test]
fn moments_table_scratch_does_not_grow_with_the_table() {
    check(2304);
    check(10 * 2304);
}
