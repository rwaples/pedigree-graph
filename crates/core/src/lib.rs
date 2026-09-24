//! pedigree-graph-core: the host-neutral Rust core of pedigree-graph (ADR 0007).
//!
//! The first resident is the row-streaming relationship engine
//! ([`relationships`]), which classifies every relationship pair up to the
//! fifth degree one individual at a time so that peak memory is linear in the
//! pedigree size (issue #11).
//!
//! Alongside it live the pairwise kinship recurrence and the depth-major
//! sweeps built beside it ([`kinship`]), the lineage counts ([`lineage`]), native
//! construction ([`graph`]), the stable depth-major
//! topological order every order-dependent kernel sweeps in ([`topology`]),
//! and the structured error enum each host maps onto its own exception
//! classes ([`error`]).

#![forbid(unsafe_code)]
// No user-reachable panics (ADR 0007): input errors are `Error` values.  A
// surviving `expect` / `unreachable!` names the internal invariant it relies
// on in an `#[expect(..., reason)]`.  `assert!` stays allowed: it states an
// invariant the caller already checked (row counts, bounds from
// `Pedigree::try_new`), and a failure is a bug the bindings surface as an
// exception rather than an abort.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]

pub mod alloc;
pub mod error;
pub mod graph;
pub mod kinship;
pub mod lineage;
pub mod pool;
pub mod relationships;
pub mod topology;

// Core types are `Send + Sync` (ADR 0007), so a host may hold them across
// threads or release its interpreter lock around a call.  This fails to
// compile if a field ever brings in `Rc`, `RefCell`, or a raw pointer.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<error::Error>();
    send_sync::<graph::PedigreeGraph>();
    send_sync::<graph::IdIndex>();
    send_sync::<graph::Columns<'static>>();
    send_sync::<graph::Limits>();
    send_sync::<kinship::matrix::Csc>();
    send_sync::<kinship::memo::PairMemo>();
    send_sync::<kinship::pairwise::KinshipPedigree<'static>>();
    send_sync::<kinship::pairwise::Walker<'static>>();
    send_sync::<lineage::ParentColumns<'static>>();
    send_sync::<relationships::Pedigree<'static>>();
    send_sync::<relationships::PedigreeColumns>();
    send_sync::<relationships::MaxDegree>();
    send_sync::<relationships::Counts>();
    send_sync::<relationships::CategorySet>();
    send_sync::<relationships::Engine<'static>>();
    send_sync::<relationships::Workspace>();
    send_sync::<relationships::WorkspacePool>();
    send_sync::<relationships::CompactView>();
    send_sync::<relationships::PairBlock>();
    send_sync::<relationships::PairBlocks>();
    send_sync::<relationships::Burden>();
};
