use std::path::PathBuf;

/// Directory where `ModelScope` models are cached.
///
/// Respects the `MODELSCOPE_CACHE` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/modelscope` (or `/tmp/.cache/modelscope` when `$HOME` is
/// unset).
#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MODELSCOPE_CACHE") {
        return PathBuf::from(dir);
    }

    std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".cache")
        .join("modelscope")
}
