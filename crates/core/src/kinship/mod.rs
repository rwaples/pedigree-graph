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
//! The kinship kernels, the inbreeding walk and the kinship sums run on the
//! pedigree's rows followed by its external founders, the external parent
//! ids that two or more rows name ([`ExternalFounders`]), and return the
//! rows' values only.  Equivalent generations and the founder means read
//! represented parents alone.
//!
//! Beside them live the sweeps that share the DP's stable depth-major order
//! (`depth_order`) but not its values: the Meuwissen-Luo inbreeding walk
//! ([`inbreeding()`]) and the Ne prerequisites of [`generations`], among
//! them the per-cohort kinship sums ([`generation_kinship_sums`]), which
//! read no kinship rows.

pub(crate) mod ancestry;
pub(crate) mod depth_order;
pub mod external;
pub mod generations;
pub mod inbreeding;
pub mod matrix;
pub mod memo;
pub mod pairwise;
pub mod rows;

pub use external::ExternalFounders;
pub use generations::{
    equivalent_generations, founder_contribution_means, generation_kinship_sums,
};
pub use inbreeding::{
    inbreeding, inbreeding_and_ancestor_counts, inbreeding_and_ancestor_counts_in,
};
pub use matrix::{approximate_kinship_csc, kinship_csc, kinship_csc_upper, Csc, MAX_CSC_NNZ};
pub use pairwise::{
    check_pair_rows, pair_kinship, support_values, view_pair_kinship, KinshipIndex, KinshipPedigree,
};
