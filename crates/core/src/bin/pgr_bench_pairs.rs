//! Benchmark driver for the slice 12 pair executions.  Internal; prints
//! digests, never pairs.
//!
//! Usage: `pgr-bench-pairs <inputs.tsv> --execution speed|memory
//! [--max-degree D] [--threads T] [--view <view.tsv>] [--dump <dir>]`
//!
//! `inputs.tsv` is the engine dump of `PedigreeColumns::read_tsv`.  `view.tsv`
//! holds one int32 view row per graph row, `-1` unselected, no header.
//! Prints one JSON object with the wall time of the engine call, the
//! process's peak RSS before and after it (`VmHWM`), and per-category pair
//! counts and digests (`PairBlock::digest`).  With `--dump`, also writes
//! every block as `<code>.first.u32` and `<code>.second.u32`, little-endian,
//! for an element-for-element comparison by the qualification driver.

use pedigree_graph_core::relationships::{
    pair_blocks, Category, CategorySet, Execution, MaxDegree, PedigreeColumns,
};
use std::path::Path;
use std::time::Instant;

fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn read_view(path: &Path) -> Vec<i32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .map(|l| {
            l.trim()
                .parse()
                .unwrap_or_else(|e| panic!("view {}: {e}", path.display()))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let mut execution = None;
    let mut max_degree = MaxDegree::MAX;
    let mut threads: usize = 1;
    let mut view_path = None;
    let mut dump = None;
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| {
            args.get(i + 1)
                .unwrap_or_else(|| panic!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--execution" => {
                execution = Some(value(i).clone());
                i += 2;
            }
            "--max-degree" => {
                max_degree = MaxDegree::try_new(value(i).parse().expect("--max-degree"))
                    .unwrap_or_else(|e| panic!("{e}"));
                i += 2;
            }
            "--threads" => {
                threads = value(i).parse().expect("--threads");
                i += 2;
            }
            "--view" => {
                view_path = Some(value(i).clone());
                i += 2;
            }
            "--dump" => {
                dump = Some(value(i).clone());
                i += 2;
            }
            other => {
                path = Some(other.to_string());
                i += 1;
            }
        }
    }
    let path = path.expect("usage: pgr-bench-pairs <inputs.tsv> --execution E [...]");
    let execution = execution.as_deref().expect("--execution is required");
    let execution =
        Execution::parse(execution).unwrap_or_else(|| panic!("unknown execution {execution}"));

    let ped = PedigreeColumns::read_tsv(Path::new(&path)).unwrap_or_else(|e| panic!("{e}"));
    let view = view_path.map(|p| read_view(Path::new(&p)));
    let n = ped.mother.len();
    let baseline_mib = peak_rss_mib();
    let pool = pedigree_graph_core::pool::configure(
        std::num::NonZeroUsize::new(threads).expect("threads must be at least 1"),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let requested = CategorySet::up_to_degree(max_degree.get());
    let t0 = Instant::now();
    let blocks = pool
        .install(|| {
            pair_blocks(
                &ped.try_borrow().expect("pedigree columns"),
                max_degree,
                requested,
                view.as_deref(),
                execution,
            )
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let seconds = t0.elapsed().as_secs_f64();
    let peak_mib = peak_rss_mib();
    eprintln!(
        "{n} rows, {} pairs in {seconds:.3}s on {threads} thread(s); RSS {baseline_mib:.0} -> {peak_mib:.0} MiB",
        blocks.total()
    );
    if let Some(dir) = dump {
        let dir = Path::new(&dir);
        std::fs::create_dir_all(dir).expect("dump dir");
        for cat in Category::ALL {
            let b = blocks.get(cat);
            for (side, values) in [("first", &b.first), ("second", &b.second)] {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{}.{side}.u32", cat.code())), bytes)
                    .expect("dump");
            }
        }
    }
    let body: Vec<String> = Category::ALL
        .iter()
        .map(|&c| {
            let b = blocks.get(c);
            format!("\"{}\": [{}, {}]", c.code(), b.len(), b.digest())
        })
        .collect();
    println!(
        "{{\"n\": {n}, \"execution\": \"{}\", \"threads\": {threads}, \"max_degree\": {}, \"view\": {}, \"seconds\": {seconds:.3}, \"baseline_rss_mib\": {baseline_mib:.1}, \"peak_rss_mib\": {peak_mib:.1}, \"pairs\": {}, \"blocks\": {{{}}}}}",
        execution.name(),
        max_degree.get(),
        view.is_some(),
        blocks.total(),
        body.join(", ")
    );
}
