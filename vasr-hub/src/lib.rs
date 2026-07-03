//! Model cache and download helpers for vASR.

mod cache;
pub mod modelscope;

pub use cache::{cache_dir, set_cache_dir};

/// Backward-compatible access to the `ModelScope` download module.
pub use modelscope::download;

/// Backward-compatible name for the model cache directory.
#[must_use]
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    cache_dir()
}
