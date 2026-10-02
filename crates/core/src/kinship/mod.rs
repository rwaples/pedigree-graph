//! Pedigree-expected kinship, the pinned float32 recurrence of ADR 0009.
//!
//! [`pair_kinship`] evaluates requested pairs and [`support_values`] fills a
//! symmetric CSC support; both walk the same memoised recurrence through
//! [`pairwise::Walker`], one memo per pool worker, nothing retained (ADR
//! 0007, 0016).
//! The matrix products ([`kinship_csc`], [`kinship_csc_upper`],
//! [`approximate_kinship_csc`]) run the depth-major DP of [`matrix`] over
//! the row storage of [`rows`]; every entry they produce is the bit the
//! pairwise walk returns for the same pair.
//!
//! Beside them live the sweeps that share the DP's stable depth-major order
//! ([`depth_order`]) but not its values: the Meuwissen-Luo inbreeding walk
//! ([`inbreeding()`]) and the Ne prerequisites of [`generations`], among
//! them the per-cohort kinship sums ([`generation_kinship_sums`]), which
//! read no kinship rows.

pub mod ancestry;
pub(crate) mod depth_order;
pub mod generations;
pub mod inbreeding;
pub mod matrix;
pub mod memo;
pub mod pairwise;
pub mod rows;

pub use generations::{
    equivalent_generations, founder_contribution_means, generation_kinship_sums,
};
pub use inbreeding::inbreeding;
pub use matrix::{approximate_kinship_csc, kinship_csc, kinship_csc_upper, Csc, MAX_CSC_NNZ};
pub use pairwise::{pair_kinship, support_values, KinshipPedigree};
