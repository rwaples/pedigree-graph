//! Count closest-category relationship pairs from a TSV dump of engine inputs.
//!
//! Usage: `pgr-count <inputs.tsv> [--max-degree D] [--threads T]`
//!
//! The TSV has a header and the columns `mother father twin orig_mother
//! orig_father` (see `tests/parity/dump_relationship_inputs.py`).  Threads
//! default to `PEDIGREE_GRAPH_THREADS`, then 1.  Prints one JSON object with
//! `n`, `threads`, `seconds`, and per-code `counts` to stdout.

use pedigree_graph_core::relationships::{count_pairs, Category, MaxDegree, PedigreeColumns};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let mut max_degree = MaxDegree::MAX;
    let mut threads: usize = std::env::var("PEDIGREE_GRAPH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            flag @ ("--max-degree" | "--threads") => {
                let value = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("{flag} needs a value"));
                match flag {
                    "--max-degree" => {
                        max_degree = MaxDegree::try_new(value.parse().expect("--max-degree"))
                            .unwrap_or_else(|e| panic!("{e}"))
                    }
                    _ => threads = value.parse().expect("--threads"),
                }
                i += 2;
            }
            other => {
                path = Some(other.to_string());
                i += 1;
            }
        }
    }
    let path = path.expect("usage: pgr-count <inputs.tsv> [--max-degree D] [--threads T]");

    let t_read = Instant::now();
    let ped =
        PedigreeColumns::read_tsv(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "read {} rows in {:.3}s",
        ped.mother.len(),
        t_read.elapsed().as_secs_f64()
    );

    let pool = pedigree_graph_core::pool::configure(
        std::num::NonZeroUsize::new(threads).expect("threads must be at least 1"),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let counts = pool
        .install(|| {
            count_pairs(
                &ped.try_borrow().expect("pedigree columns"),
                max_degree,
                None,
            )
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let seconds = t0.elapsed().as_secs_f64();
    eprintln!("counted in {seconds:.3}s on {threads} thread(s)");

    let body: Vec<String> = Category::ALL
        .iter()
        .map(|&c| format!("\"{}\": {}", c.code(), counts.get(c)))
        .collect();
    println!(
        "{{\"n\": {}, \"threads\": {}, \"max_degree\": {}, \"seconds\": {:.3}, \"counts\": {{{}}}}}",
        ped.mother.len(),
        threads,
        max_degree.get(),
        seconds,
        body.join(", ")
    );
}
