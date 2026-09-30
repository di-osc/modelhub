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

mod cache;
mod huggingface;
mod modelscope;
mod ops;
mod repos;
mod unified;
mod upload;

/// High-level operations, one per CLI subcommand.
pub use ops::{
    CheckOptions, ClearOptions, ClearSummary, DownloadOptions, ListOptions, RepoEntry, check,
    clear, download, list, upload,
};

/// Types that appear in the operation signatures and result structs.
pub use repos::{CacheSource, RepoHit, RepoKind, RepoStatus};

/// Result of [`download`].
pub use unified::DownloadedRepo;

/// Types for [`upload`].
pub use upload::{BackendUpload, UploadBackend, UploadCounts, UploadOptions, UploadSummary};
