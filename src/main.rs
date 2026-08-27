use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "modelhub",
    version,
    about = "Download models from supported model hubs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Detect every supported backend, download the model once, and link all caches.
    Download {
        /// Model identifier, for example `org/name`.
        model_id: String,
        /// Revision to request. Defaults to `master` on `ModelScope` and `main` on Hugging Face.
        #[arg(short, long)]
        revision: Option<String>,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Maximum number of files downloaded concurrently.
        #[arg(short = 'j', long, default_value_t = 4, value_parser = parse_jobs)]
        jobs: usize,
        /// Keep both backend versions even when model weights differ.
        #[arg(long)]
        all_backends: bool,
    },
    /// List models stored in the modelhub cache.
    List {
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Remove one model or all models from the modelhub cache.
    Clear {
        /// Model identifier to remove, for example `org/name`.
        #[arg(required_unless_present = "all", conflicts_with = "all")]
        model_id: Option<String>,
        /// Remove the entire modelhub cache.
        #[arg(long)]
        all: bool,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

const MODEL_ID_FILE: &str = ".modelhub-model-id";

fn parse_jobs(value: &str) -> Result<usize, String> {
    let jobs = value
        .parse::<usize>()
        .map_err(|_| "jobs must be a positive integer".to_owned())?;
    if jobs == 0 {
        return Err("jobs must be at least 1".to_owned());
    }
    Ok(jobs)
}

fn link_directory(source: &Path, target: &Path) -> anyhow::Result<()> {
    let source = fs::canonicalize(source)
        .with_context(|| format!("modelhub cache path does not exist: {}", source.display()))?;
    if fs::canonicalize(target).is_ok_and(|existing| existing == source) {
        eprintln!("Warning: cache link already exists: {}", target.display());
        return Ok(());
    }
    if let Ok(metadata) = fs::symlink_metadata(target) {
        if metadata.file_type().is_symlink() {
            if fs::canonicalize(target).is_ok_and(|existing| existing == source) {
                eprintln!("Warning: cache link already exists: {}", target.display());
                return Ok(());
            }
            eprintln!(
                "Warning: preserving existing backend link: {}",
                target.display()
            );
            return Ok(());
        } else if metadata.is_dir() {
            if fs::read_dir(target)?.next().is_none() {
                // An empty backend-created directory can safely be replaced by the link.
                fs::remove_dir(target)?;
            } else {
                // Preserve an existing native cache and add links for entries it does not have.
                eprintln!(
                    "Warning: backend cache already exists; preserving and reusing it: {}",
                    target.display()
                );
                merge_directory(&source, target)?;
                return Ok(());
            }
        } else if fs::canonicalize(target).is_ok_and(|existing| existing == source) {
            return Ok(());
        } else {
            eprintln!(
                "Warning: preserving existing backend cache path: {}",
                target.display()
            );
            return Ok(());
        }
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    create_symlink(&source, target).with_context(|| {
        format!(
            "failed to link {} -> {}",
            target.display(),
            source.display()
        )
    })
}

fn merge_directory(source: &Path, target: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if entry.file_name() == MODEL_ID_FILE {
            continue;
        }
        if let Ok(metadata) = fs::symlink_metadata(&target_path) {
            if metadata.is_dir() && source_path.is_dir() {
                merge_directory(&source_path, &target_path)?;
            }
            continue;
        }
        if source_path.is_dir() {
            create_symlink(&source_path, &target_path)?;
        } else {
            fs::copy(&source_path, &target_path)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn create_symlink(source: &Path, target: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, target)
}

#[cfg(windows)]
fn create_symlink(source: &Path, target: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(source, target)
}

#[cfg(not(any(unix, windows)))]
fn create_symlink(_source: &Path, _target: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "directory links are unsupported on this platform",
    ))
}

#[derive(Debug)]
struct CachedModel {
    model_id: String,
    paths: Vec<PathBuf>,
}

fn model_id_from_dir(path: &Path, huggingface_layout: bool) -> Option<String> {
    let marker = path.join(MODEL_ID_FILE);
    if let Ok(model_id) = fs::read_to_string(marker) {
        let model_id = model_id.trim();
        if !model_id.is_empty() {
            return Some(model_id.to_owned());
        }
    }
    let mut name = path.file_name()?.to_str()?;
    if huggingface_layout {
        name = name.strip_prefix("models--")?;
    }
    let (namespace, model) = name.split_once("--")?;
    Some(format!("{namespace}/{model}"))
}

fn collect_cache_parent(
    models: &mut BTreeMap<String, Vec<PathBuf>>,
    parent: &Path,
    huggingface_layout: bool,
) -> anyhow::Result<()> {
    if !parent.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(parent)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(model_id) = model_id_from_dir(&path, huggingface_layout) {
            models.entry(model_id).or_default().push(path);
        }
    }
    Ok(())
}

fn cached_models(cache_root: &Path) -> anyhow::Result<Vec<CachedModel>> {
    let mut models = BTreeMap::new();
    collect_cache_parent(&mut models, &cache_root.join("models"), false)?;
    // Include layouts created by earlier development versions.
    collect_cache_parent(
        &mut models,
        &cache_root.join("modelscope").join("models"),
        false,
    )?;
    collect_cache_parent(
        &mut models,
        &cache_root.join("huggingface").join("hub"),
        true,
    )?;
    Ok(models
        .into_iter()
        .map(|(model_id, paths)| CachedModel { model_id, paths })
        .collect())
}

fn directory_size(path: &Path) -> anyhow::Result<u64> {
    let mut size = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_file() {
            size += metadata.len();
        } else if metadata.is_dir() {
            size += directory_size(&entry.path())?;
        }
    }
    Ok(size)
}

fn human_size(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit = 0;
    let mut divisor = 1u64;
    while size / divisor >= 1024 && unit < UNITS.len() - 1 {
        divisor *= 1024;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        let whole = size / divisor;
        let decimal = (size % divisor) * 10 / divisor;
        format!("{whole}.{decimal} {}", UNITS[unit])
    }
}

fn list(cache_root: &Path) -> anyhow::Result<()> {
    let models = cached_models(cache_root)?;
    if models.is_empty() {
        println!("No cached models in {}", cache_root.display());
        return Ok(());
    }
    println!("{:<48} {:>12}  PATH", "MODEL", "SIZE");
    for model in models {
        let size = model.paths.iter().try_fold(0u64, |total, path| {
            directory_size(path).map(|size| total + size)
        })?;
        println!(
            "{:<48} {:>12}  {}",
            model.model_id,
            human_size(size),
            model.paths[0].display()
        );
    }
    Ok(())
}

fn backend_cache_paths(model_id: &str) -> [PathBuf; 2] {
    [
        modelscope_cache_path(model_id),
        huggingface_cache_path(model_id),
    ]
}

fn modelscope_cache_path(model_id: &str) -> PathBuf {
    modelhub::modelscope::cache_dir()
        .join("models")
        .join(model_id.replace('/', "--"))
}

fn huggingface_cache_path(model_id: &str) -> PathBuf {
    modelhub::huggingface::cache_dir().join(format!("models--{}", model_id.replace('/', "--")))
}

fn remove_links_into(
    path: &Path,
    cache_root: &Path,
    model_sources: &[PathBuf],
) -> anyhow::Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.file_type().is_symlink() {
        if fs::canonicalize(path)
            .is_ok_and(|target| target.starts_with(cache_root) || model_sources.contains(&target))
        {
            fs::remove_file(path)?;
            eprintln!("Removed backend cache link {}", path.display());
        }
        return Ok(());
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            remove_links_into(&entry?.path(), cache_root, model_sources)?;
        }
    }
    Ok(())
}

fn clear_model(cache_root: &Path, model: &CachedModel) -> anyhow::Result<()> {
    let canonical_root = fs::canonicalize(cache_root).unwrap_or_else(|_| cache_root.to_path_buf());
    let model_sources: Vec<_> = model
        .paths
        .iter()
        .filter_map(|path| fs::canonicalize(path).ok())
        .collect();
    for backend_path in backend_cache_paths(&model.model_id) {
        remove_links_into(&backend_path, &canonical_root, &model_sources)?;
    }
    for path in &model.paths {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() {
            fs::remove_file(path)?;
        } else {
            fs::remove_dir_all(path)?;
        }
        eprintln!("Removed cached model {}", path.display());
    }
    garbage_collect_blobs(cache_root)?;
    Ok(())
}

fn garbage_collect_blobs(cache_root: &Path) -> anyhow::Result<()> {
    let blobs = cache_root.join("blobs").join("sha256");
    if !blobs.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(blobs)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if fs::metadata(&path)?.nlink() <= 1 {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn clear(cache_root: &Path, model_id: Option<&str>, all: bool) -> anyhow::Result<()> {
    if all {
        let absolute = if cache_root.exists() {
            fs::canonicalize(cache_root)?
        } else if cache_root.is_absolute() {
            cache_root.to_path_buf()
        } else {
            std::env::current_dir()?.join(cache_root)
        };
        if absolute.parent().is_none()
            || absolute == std::env::current_dir()?
            || std::env::var("HOME").is_ok_and(|home| absolute == Path::new(&home))
            || absolute.components().count() < 3
        {
            bail!("refusing to clear unsafe cache root {}", absolute.display());
        }
    }
    let models = cached_models(cache_root)?;
    if all {
        for model in &models {
            clear_model(cache_root, model)?;
        }
        if cache_root.exists() {
            fs::remove_dir_all(cache_root)
                .with_context(|| format!("failed to clear {}", cache_root.display()))?;
        }
        println!("Cleared all modelhub cache from {}", cache_root.display());
        return Ok(());
    }
    let model_id = model_id.context("model ID is required unless --all is used")?;
    let Some(model) = models.iter().find(|model| model.model_id == model_id) else {
        println!(
            "Model `{model_id}` is not cached in {}",
            cache_root.display()
        );
        return Ok(());
    };
    clear_model(cache_root, model)?;
    println!("Cleared `{model_id}` from the modelhub cache");
    Ok(())
}

async fn download(
    model_id: &str,
    revision: Option<&str>,
    cache_root: &Path,
    jobs: usize,
    all_backends: bool,
) -> anyhow::Result<()> {
    fs::create_dir_all(cache_root)?;
    let downloaded = modelhub::unified::download_model(
        model_id,
        revision.unwrap_or("main"),
        revision.unwrap_or("master"),
        cache_root,
        jobs,
        all_backends,
    )
    .await?;
    if let Some(root) = downloaded.huggingface_root {
        link_directory(&root, &huggingface_cache_path(model_id))?;
    }
    if let Some(root) = downloaded.modelscope_root {
        link_directory(&root, &modelscope_cache_path(model_id))?;
    }
    eprintln!("Verified backend manifests and linked compatible cache snapshots.");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Download {
            model_id,
            revision,
            cache_dir,
            jobs,
            all_backends,
        } => {
            let cache_dir = cache_dir.unwrap_or_else(modelhub::cache::cache_dir);
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(download(
                &model_id,
                revision.as_deref(),
                &cache_dir,
                jobs,
                all_backends,
            ))
        }
        Command::List { cache_dir } => {
            let cache_dir = cache_dir.unwrap_or_else(modelhub::cache::cache_dir);
            list(&cache_dir)
        }
        Command::Clear {
            model_id,
            all,
            cache_dir,
        } => {
            let cache_dir = cache_dir.unwrap_or_else(modelhub::cache::cache_dir);
            clear(&cache_dir, model_id.as_deref(), all)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, human_size, model_id_from_dir};
    use clap::Parser;
    use std::path::Path;

    #[test]
    fn decodes_cache_directory_names() {
        assert_eq!(
            model_id_from_dir(Path::new("/cache/acme--demo--v2"), false).as_deref(),
            Some("acme/demo--v2")
        );
        assert_eq!(
            model_id_from_dir(Path::new("/cache/models--acme--demo"), true).as_deref(),
            Some("acme/demo")
        );
    }

    #[test]
    fn formats_cache_sizes() {
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(1536), "1.5 KiB");
    }

    #[test]
    fn clear_requires_a_model_or_all() {
        assert!(Cli::try_parse_from(["modelhub", "clear"]).is_err());
        assert!(Cli::try_parse_from(["modelhub", "clear", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "clear", "--all"]).is_ok());
    }

    #[test]
    fn download_jobs_must_be_positive() {
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "8"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "0"]).is_err());
    }
}
