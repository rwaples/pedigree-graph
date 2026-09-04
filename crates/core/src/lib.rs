//! pedigree-graph-core: the host-neutral Rust core of pedigree-graph (ADR 0007).
//!
//! The first resident is the row-streaming relationship engine
//! ([`relationships`]), which classifies every relationship pair up to the
//! fifth degree one individual at a time so that peak memory is linear in the
//! pedigree size (issue #11).

#![forbid(unsafe_code)]

pub mod relationships;
