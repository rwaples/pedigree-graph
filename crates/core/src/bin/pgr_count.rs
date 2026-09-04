//! Count relationship pairs from a TSV dump of engine inputs.
//!
//! Usage: `pgr-count <inputs.tsv> [--max-degree D] [--threads T]`
//!
//! The TSV has a header and the columns `mother father twin orig_mother
//! orig_father` (see `tests/parity/dump_relationship_inputs.py`).  Threads
//! default to `PEDIGREE_GRAPH_THREADS`, then 1.  Prints one JSON object with
//! `n`, `threads`, `seconds`, and per-code `counts` to stdout.

use pedigree_graph_core::relationships::{count_pairs, Category, Pedigree};
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn read_tsv(path: &str) -> Pedigree {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut ped = Pedigree::default();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.expect("read line");
        if line_no == 0 {
            continue;
        }
        let mut fields = line.split('\t').map(|s| {
            s.parse::<i64>()
                .unwrap_or_else(|_| panic!("line {line_no}: {s:?}"))
        });
        let mut next = || {
            fields
                .next()
                .unwrap_or_else(|| panic!("line {line_no}: too few columns"))
        };
        ped.mother.push(next() as i32);
        ped.father.push(next() as i32);
        ped.twin.push(next() as i32);
        ped.orig_mother.push(next());
        ped.orig_father.push(next());
    }
    ped
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let mut max_degree = 5u8;
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
                    "--max-degree" => max_degree = value.parse().expect("--max-degree"),
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
    let ped = read_tsv(&path);
    eprintln!(
        "read {} rows in {:.3}s",
        ped.len(),
        t_read.elapsed().as_secs_f64()
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let t0 = Instant::now();
    let counts = pool.install(|| count_pairs(&ped, max_degree));
    let seconds = t0.elapsed().as_secs_f64();
    eprintln!("counted in {seconds:.3}s on {threads} thread(s)");

    let body: Vec<String> = Category::ALL
        .iter()
        .map(|&c| format!("\"{}\": {}", c.code(), counts.get(c)))
        .collect();
    println!(
        "{{\"n\": {}, \"threads\": {}, \"max_degree\": {}, \"seconds\": {:.3}, \"counts\": {{{}}}}}",
        ped.len(),
        threads,
        max_degree,
        seconds,
        body.join(", ")
    );
}
