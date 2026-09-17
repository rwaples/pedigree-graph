//! Bit-for-bit parity with `PedigreeGraph.relationship_counts` on the dumped fixtures.
//!
//! `tests/fixtures/<name>.tsv` is written by
//! `tests/parity/dump_relationship_inputs.py` (frozen, 0.7.1 API) and
//! `<name>.counts.json` by `tests/parity/dump_relationship_counts.py` from
//! `relationship_counts(max_degree=5)` on a graph rebuilt from that TSV.
//! Every fixture must match on all 23 categories, at one thread and at four.

use pedigree_graph_core::relationships::{
    count_pairs, Category, Counts, MaxDegree, PedigreeColumns,
};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Pull `"<code>": <int>` pairs out of the counts JSON without a JSON crate.
fn read_counts(path: &Path) -> Counts {
    let text = std::fs::read_to_string(path).unwrap();
    let mut counts = Counts::default();
    for cat in Category::ALL {
        let needle = format!("\"{}\":", cat.code());
        let at = text
            .find(&needle)
            .unwrap_or_else(|| panic!("{} missing {}", path.display(), cat.code()));
        let rest = text[at + needle.len()..].trim_start();
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        counts.add(cat, digits.parse().unwrap());
    }
    counts
}

fn run_all(threads: usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let mut names: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no fixtures dumped under {}",
        fixtures_dir().display()
    );
    let mut failures = Vec::new();
    for tsv in &names {
        let name = tsv.file_stem().unwrap().to_string_lossy().to_string();
        let expected = read_counts(&tsv.with_extension("counts.json"));
        let ped = PedigreeColumns::read_tsv(tsv).unwrap();
        let got = pool
            .install(|| count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None))
            .unwrap();
        for cat in Category::ALL {
            if got.get(cat) != expected.get(cat) {
                failures.push(format!(
                    "{name}: {} got {} expected {}",
                    cat.code(),
                    got.get(cat),
                    expected.get(cat)
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn counts_match_relationship_counts_single_thread() {
    run_all(1);
}

#[test]
fn counts_match_relationship_counts_four_threads() {
    run_all(4);
}
