//! Pedigree-expected kinship, the pinned float32 recurrence of ADR 0009.
//!
//! [`pair_kinship`] evaluates requested pairs and [`support_values`] fills a
//! symmetric CSC support; both walk the same memoised recurrence through
//! [`pairwise::Walker`], one memo per call, nothing retained (ADR 0007).  The
//! memo layout is chosen by [`Layout`] while slice 13's bake-off runs; the
//! loser is deleted once `docs/pedigree-graph-0.8-migration/gate/13a` records
//! the selection.

pub mod memo;
pub mod pairwise;

pub use memo::Layout;
pub use pairwise::{pair_kinship, support_values, KinshipPedigree};
