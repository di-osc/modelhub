//! Model cache and download helpers.

pub mod modelscope;

/// Backward-compatible alias for the `ModelScope` cache directory.
pub use modelscope::{cache_dir, set_cache_dir};

/// Backward-compatible access to the `ModelScope` download module.
pub use modelscope::download;

/// Backward-compatible name for the model cache directory.
#[must_use]
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    modelscope::cache_dir()
}
