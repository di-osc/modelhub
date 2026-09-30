//! Model cache and download helpers.
//!
//! High-level operations mirror the CLI subcommands and return structured data:
//!
//! ```no_run
//! use modelhub::{ListOptions, RepoStatus};
//!
//! # fn main() -> anyhow::Result<()> {
//! let entries = modelhub::list(&ListOptions::default())?;
//! for entry in &entries {
//!     let status = entry.status.unwrap_or(RepoStatus::Unknown);
//!     println!("{}\t{status}", entry.id);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! `download` and `check` are `async`; `list` and `clear` are synchronous.

pub mod cache;
pub mod huggingface;
pub mod modelscope;
pub mod ops;
pub mod repos;
pub mod unified;

/// Backward-compatible alias for the `ModelScope` cache directory.
pub use modelscope::{cache_dir, set_cache_dir};

/// Backward-compatible access to the `ModelScope` download module.
pub use modelscope::download;

/// High-level operations, one per CLI subcommand.
pub use ops::{
    CheckOptions, ClearOptions, ClearSummary, DownloadOptions, ListOptions, RepoEntry, check,
    clear, download, list,
};

/// Types shared by the discovery and operations APIs.
pub use repos::{CacheSource, CachedRepo, RepoHit, RepoKind, RepoStatus};

/// Backward-compatible name for the model cache directory.
#[must_use]
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    modelscope::cache_dir()
}
