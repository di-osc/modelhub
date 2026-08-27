use std::path::PathBuf;

/// Return the root directory owned by `modelhub` itself.
///
/// `MODELHUB_CACHE` takes precedence; otherwise this defaults to
/// `$HOME/.cache/modelhub` (or `/tmp/.cache/modelhub` when `HOME` is unset).
#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MODELHUB_CACHE") {
        return PathBuf::from(dir);
    }

    std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".cache")
        .join("modelhub")
}
