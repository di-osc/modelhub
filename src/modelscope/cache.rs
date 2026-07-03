use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

static CACHE_DIR_OVERRIDE: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();

/// Override the `ModelScope` cache directory for the current process.
pub fn set_cache_dir(dir: impl Into<PathBuf>) {
    let lock = CACHE_DIR_OVERRIDE.get_or_init(|| RwLock::new(None));
    if let Ok(mut value) = lock.write() {
        *value = Some(dir.into());
    }
}

/// Directory where `ModelScope` models are cached.
///
/// Respects the `MODELSCOPE_CACHE` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/modelscope` (or `/tmp/.cache/modelscope` when `$HOME` is
/// unset).
#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Some(lock) = CACHE_DIR_OVERRIDE.get()
        && let Ok(value) = lock.read()
        && let Some(dir) = value.as_ref()
    {
        return dir.clone();
    }

    if let Ok(dir) = std::env::var("MODELSCOPE_CACHE") {
        return PathBuf::from(dir);
    }

    std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".cache")
        .join("modelscope")
}
