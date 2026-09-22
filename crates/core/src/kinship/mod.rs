//! Pedigree-expected kinship, the pinned float32 recurrence of ADR 0009.
//!
//! [`pair_kinship`] evaluates requested pairs and [`support_values`] fills a
//! symmetric CSC support; both walk the same memoised recurrence through
//! [`pairwise::Walker`], one memo per call, nothing retained (ADR 0007).

pub mod memo;
pub mod pairwise;

pub use pairwise::{pair_kinship, support_values, KinshipPedigree};
