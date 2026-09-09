//! pedigree-graph-core: the host-neutral Rust core of pedigree-graph (ADR 0007).
//!
//! The first resident is the row-streaming relationship engine
//! ([`relationships`]), which classifies every relationship pair up to the
//! fifth degree one individual at a time so that peak memory is linear in the
//! pedigree size (issue #11).
//!
//! Alongside it live the stable depth-major topological order every
//! order-dependent kernel sweeps in ([`topology`]) and the structured error
//! enum each host maps onto its own exception classes ([`error`]).

#![forbid(unsafe_code)]

pub mod error;
pub mod relationships;
pub mod topology;
