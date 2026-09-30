//! Hugging Face hub cache location.

use std::path::PathBuf;

/// Return the cache directory used by the Hugging Face Hub client.
///
/// `HUGGINGFACE_HUB_CACHE` wins. Otherwise this is `$HF_HOME/hub`, or
/// `$HOME/.cache/huggingface/hub` when `HF_HOME` is unset (`/tmp` when
/// `HOME` is unset).
#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}
