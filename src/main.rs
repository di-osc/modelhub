use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, CellAlignment, ContentArrangement, Table};
use std::collections::{BTreeMap, BTreeSet, HashSet};
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
    /// List models cached by modelhub and by every supported backend.
    List {
        /// Modelhub cache to scan (defaults to `$HOME/.cache/modelhub`).
        /// `ModelScope` and Hugging Face caches are scanned as well.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Remove one model from the modelhub cache, or every model with `--all`.
    Clear {
        /// Model identifier to remove, for example `org/name`.
        /// Without `--all` or `--backend`, only the modelhub copy is removed.
        #[arg(required_unless_present = "all")]
        model_id: Option<String>,
        /// Remove from every supported backend. With a model id, only that model is removed.
        /// Without a model id, every model directory is removed. Datasets are kept.
        #[arg(long)]
        all: bool,
        /// Only touch this backend: `modelhub`, `modelscope`, or `huggingface`.
        #[arg(long, value_parser = parse_backend)]
        backend: Option<CacheSource>,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

const MODEL_ID_FILE: &str = ".modelhub-model-id";

/// Parse `--backend` as one supported cache.
fn parse_backend(value: &str) -> Result<CacheSource, String> {
    match value {
        "huggingface" => Ok(CacheSource::HuggingFace),
        "modelhub" => Ok(CacheSource::ModelHub),
        "modelscope" => Ok(CacheSource::ModelScope),
        _ => Err("backend must be modelhub, modelscope, or huggingface".to_owned()),
    }
}

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

/// Cache that contributed a discovered model directory.
///
/// Variant order is alphabetical so a `BTreeSet` prints stable backend labels.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CacheSource {
    HuggingFace,
    ModelHub,
    ModelScope,
}

impl CacheSource {
    /// Column label for this cache.
    const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelHub => "modelhub",
            Self::ModelScope => "modelscope",
        }
    }
}

/// One on-disk directory that belongs to a model.
#[derive(Debug)]
struct CacheHit {
    source: CacheSource,
    path: PathBuf,
}

/// A model found in one or more caches.
#[derive(Debug)]
struct ListedModel {
    model_id: String,
    hits: Vec<CacheHit>,
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

/// Record model directories directly under `parent`.
///
/// `huggingface_layout` decodes `models--org--name` names used by the Hugging Face
/// hub. Other caches use `org--name`. A symlink to a directory is included, so a
/// backend link into the modelhub cache still shows up under that backend.
fn collect_cache_parent(
    models: &mut BTreeMap<String, Vec<CacheHit>>,
    parent: &Path,
    huggingface_layout: bool,
    source: CacheSource,
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
            remember(models, model_id, CacheHit { source, path });
        }
    }
    Ok(())
}

/// Keep the first hit when the same source already recorded this directory.
fn remember(models: &mut BTreeMap<String, Vec<CacheHit>>, model_id: String, hit: CacheHit) {
    let hits = models.entry(model_id).or_default();
    let duplicate = hits
        .iter()
        .any(|existing| existing.source == hit.source && same_path(&existing.path, &hit.path));
    if !duplicate {
        hits.push(hit);
    }
}

/// Compare paths by their canonical target when both can be resolved.
fn same_path(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Scan layouts owned by modelhub, including directories from older versions.
fn collect_modelhub(
    models: &mut BTreeMap<String, Vec<CacheHit>>,
    cache_root: &Path,
) -> anyhow::Result<()> {
    collect_cache_parent(
        models,
        &cache_root.join("models"),
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        models,
        &cache_root.join("modelscope").join("models"),
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        models,
        &cache_root.join("huggingface").join("hub"),
        true,
        CacheSource::ModelHub,
    )?;
    Ok(())
}

/// Models stored in the modelhub cache. Native backend copies are excluded so
/// `clear` does not remove caches created outside modelhub.
fn modelhub_models(cache_root: &Path) -> anyhow::Result<Vec<CachedModel>> {
    let mut models = BTreeMap::new();
    collect_modelhub(&mut models, cache_root)?;
    Ok(models
        .into_iter()
        .map(|(model_id, hits)| CachedModel {
            model_id,
            paths: hits.into_iter().map(|hit| hit.path).collect(),
        })
        .collect())
}

/// Models visible to modelhub, `ModelScope`, and Hugging Face.
///
/// `modelscope_models` is the `ModelScope` `models/` directory. `huggingface_hub`
/// is the Hugging Face hub directory that contains `models--*` folders.
fn discover_models(
    modelhub_root: &Path,
    modelscope_models: &Path,
    huggingface_hub: &Path,
) -> anyhow::Result<Vec<ListedModel>> {
    let mut models = BTreeMap::new();
    collect_modelhub(&mut models, modelhub_root)?;
    collect_cache_parent(
        &mut models,
        modelscope_models,
        false,
        CacheSource::ModelScope,
    )?;
    collect_cache_parent(&mut models, huggingface_hub, true, CacheSource::HuggingFace)?;
    Ok(models
        .into_iter()
        .map(|(model_id, hits)| ListedModel { model_id, hits })
        .collect())
}

/// Comma-separated backend labels for the hits, in stable alphabetical order.
fn source_labels(hits: &[CacheHit]) -> String {
    hits.iter()
        .map(|hit| hit.source)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(CacheSource::label)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Canonical model roots, omitting a path that already lives inside another hit.
///
/// modelhub links a backend cache entry at the snapshot directory, which sits
/// inside the modelhub model directory. Counting both would report the same
/// files twice.
fn outermost_paths(hits: &[CacheHit]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = hits
        .iter()
        .filter_map(|hit| fs::canonicalize(&hit.path).ok())
        .collect();
    roots.sort();
    roots.dedup();
    let all = roots.clone();
    roots.retain(|path| {
        !all.iter()
            .any(|other| other != path && path.starts_with(other))
    });
    roots
}

/// User-facing path for a canonical root. Prefer the modelhub directory when the
/// backend entry is only a link to that same directory.
fn representative_path(hits: &[CacheHit], root: &Path) -> PathBuf {
    let mut matches: Vec<&CacheHit> = hits
        .iter()
        .filter(|hit| fs::canonicalize(&hit.path).is_ok_and(|path| path == root))
        .collect();
    matches.sort_by_key(|hit| hit.source);
    matches
        .iter()
        .find(|hit| hit.source == CacheSource::ModelHub)
        .or_else(|| matches.first())
        .map_or_else(|| root.to_path_buf(), |hit| hit.path.clone())
}

/// Paths to print for a model: one per distinct copy on disk.
fn display_paths(hits: &[CacheHit]) -> Vec<PathBuf> {
    outermost_paths(hits)
        .into_iter()
        .map(|root| representative_path(hits, &root))
        .collect()
}

/// Record this file's inode. Returns false when the same inode was already counted.
///
/// Platforms without inode metadata always return true, so each directory entry
/// is counted on its own.
fn remember_inode(metadata: &fs::Metadata, files: &mut HashSet<(u64, u64)>) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        files.insert((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = (metadata, files);
        true
    }
}

/// Add `metadata`'s length unless this inode was already counted.
fn account_file(metadata: &fs::Metadata, files: &mut HashSet<(u64, u64)>) -> u64 {
    if !remember_inode(metadata, files) {
        return 0;
    }
    metadata.len()
}

/// Byte size of `path`, counting each file inode once.
///
/// Hugging Face snapshot entries are symlinks into `blobs/`. Following those
/// links, and skipping inodes already seen, avoids adding the same blob again
/// for every revision. Directory links that leave `path` are ignored so a stray
/// link cannot walk unrelated trees.
fn directory_size(
    path: &Path,
    files: &mut HashSet<(u64, u64)>,
    dirs: &mut HashSet<PathBuf>,
) -> anyhow::Result<u64> {
    let Ok(root) = fs::canonicalize(path) else {
        return Ok(0);
    };
    directory_size_within(&root, &root, files, dirs)
}

/// Walk `path` while staying inside `root`.
fn directory_size_within(
    root: &Path,
    path: &Path,
    files: &mut HashSet<(u64, u64)>,
    dirs: &mut HashSet<PathBuf>,
) -> anyhow::Result<u64> {
    let Ok(canonical) = fs::canonicalize(path) else {
        return Ok(0);
    };
    if !canonical.starts_with(root) || !dirs.insert(canonical.clone()) {
        return Ok(0);
    }
    let mut size = 0u64;
    for entry in fs::read_dir(canonical)? {
        let entry_path = entry?.path();
        let metadata = fs::symlink_metadata(&entry_path)?;
        size += if metadata.file_type().is_symlink() {
            linked_entry_size(root, &entry_path, files, dirs)?
        } else if metadata.is_file() {
            account_file(&metadata, files)
        } else if metadata.is_dir() {
            directory_size_within(root, &entry_path, files, dirs)?
        } else {
            0
        };
    }
    Ok(size)
}

/// Size of a symlink target. File targets are counted even when the blob lives
/// beside the model directory; directory targets must stay inside `root`.
fn linked_entry_size(
    root: &Path,
    path: &Path,
    files: &mut HashSet<(u64, u64)>,
    dirs: &mut HashSet<PathBuf>,
) -> anyhow::Result<u64> {
    match fs::metadata(path) {
        Ok(target) if target.is_file() => Ok(account_file(&target, files)),
        Ok(target) if target.is_dir() => directory_size_within(root, path, files, dirs),
        _ => Ok(0),
    }
}

/// Disk usage of every distinct copy of a model.
fn model_disk_size(hits: &[CacheHit]) -> anyhow::Result<u64> {
    let mut files = HashSet::new();
    let mut dirs = HashSet::new();
    let mut total = 0;
    for root in outermost_paths(hits) {
        total += directory_size(&root, &mut files, &mut dirs)?;
    }
    Ok(total)
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

/// Print models from the modelhub cache and from each native backend cache.
fn list(cache_root: &Path) -> anyhow::Result<()> {
    let modelscope_models = modelhub::modelscope::cache_dir().join("models");
    let huggingface_hub = modelhub::huggingface::cache_dir();
    let models = discover_models(cache_root, &modelscope_models, &huggingface_hub)?;
    if models.is_empty() {
        println!(
            "No cached models in modelhub ({}), ModelScope ({}), or Hugging Face ({})",
            cache_root.display(),
            modelhub::modelscope::cache_dir().display(),
            huggingface_hub.display()
        );
        return Ok(());
    }

    let rows = models
        .iter()
        .map(|model| {
            Ok(ListRow {
                model_id: model.model_id.clone(),
                size: human_size(model_disk_size(&model.hits)?),
                sources: source_labels(&model.hits),
                paths: display_paths(&model.hits),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    render_list(&rows);
    Ok(())
}

/// One formatted `list` row.
struct ListRow {
    model_id: String,
    size: String,
    sources: String,
    paths: Vec<PathBuf>,
}

/// Write the model list as a bordered table.
fn render_list(rows: &[ListRow]) {
    println!("{}", format_table(rows));
}

/// Render `rows` with `comfy-table`, fitting the current terminal when possible.
fn format_table(rows: &[ListRow]) -> String {
    render_model_table(rows, None)
}

/// Build the list table.
///
/// `width` overrides the terminal width so tests stay stable. `None` lets
/// `comfy-table` measure the terminal and wrap long paths instead of stretching
/// the table past the screen. Several cache paths for one model stay in the
/// same `PATH` cell, one path per line. `SIZE` is right-aligned.
fn render_model_table(rows: &[ListRow], width: Option<u16>) -> String {
    let mut table = Table::new();
    table
        .load_style(UTF8_FULL.with_rounded_corners().with_solid_inner_borders())
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["MODEL", "SIZE", "BACKENDS", "PATH"]);
    if let Some(width) = width {
        table.set_width(width);
    }
    // Column 1 is SIZE. Alignment applies to the header and every data cell.
    if let Some(column) = table.column_mut(1) {
        column.set_cell_alignment(CellAlignment::Right);
    }
    for row in rows {
        let path = row
            .paths
            .iter()
            .map(|path| display_path(path))
            .collect::<Vec<_>>()
            .join("\n");
        table.add_row(vec![
            Cell::new(row.model_id.as_str()),
            Cell::new(row.size.as_str()),
            Cell::new(row.sources.as_str()),
            Cell::new(path),
        ]);
    }
    table.to_string()
}

/// Show `$HOME` as `~` so repeated cache prefixes do not dominate the table.
fn display_path(path: &Path) -> String {
    shorten_home(
        &path.display().to_string(),
        std::env::var("HOME").ok().as_deref(),
    )
}

/// Replace a `home` prefix with `~`.
fn shorten_home(path: &str, home: Option<&str>) -> String {
    if let Some(home) = home.filter(|home| !home.is_empty())
        && let Some(rest) = path.strip_prefix(&format!("{home}/"))
    {
        return format!("~/{rest}");
    }
    path.to_owned()
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

/// Backends selected by `--all` and `--backend`.
///
/// A named backend limits the command to that cache. `--all` without
/// `--backend` covers every supported cache. A model id without `--all` or
/// `--backend` stays on modelhub.
fn clear_targets(all: bool, backend: Option<CacheSource>) -> BTreeSet<CacheSource> {
    backend.map_or_else(
        || {
            if all {
                BTreeSet::from([
                    CacheSource::HuggingFace,
                    CacheSource::ModelHub,
                    CacheSource::ModelScope,
                ])
            } else {
                BTreeSet::from([CacheSource::ModelHub])
            }
        },
        |backend| BTreeSet::from([backend]),
    )
}

/// Refuse to delete a cache root that is `$HOME`, the working directory, or too shallow.
fn refuse_unsafe_cache_root(cache_root: &Path) -> anyhow::Result<()> {
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
    Ok(())
}

/// Absolute path of `path` without following a final symlink.
fn removal_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let Some(name) = path.file_name() else {
        return path.to_path_buf();
    };
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::canonicalize(parent)
        .unwrap_or_else(|_| parent.to_path_buf())
        .join(name)
}

/// Refuse to delete `$HOME`, the working directory, or a path with too few components.
fn refuse_unsafe_removal(path: &Path) -> anyhow::Result<()> {
    let absolute = removal_path(path);
    if absolute.parent().is_none()
        || std::env::current_dir().is_ok_and(|cwd| absolute == cwd)
        || std::env::var("HOME").is_ok_and(|home| absolute == Path::new(&home))
        || absolute.components().count() < 4
    {
        bail!("refusing to clear unsafe path {}", absolute.display());
    }
    Ok(())
}

/// Delete a file or directory. A symlink is removed without following it.
fn remove_tree(path: &Path) -> anyhow::Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    refuse_unsafe_removal(path)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path)?;
    } else if metadata.is_dir() {
        fs::remove_dir_all(path)?;
    }
    Ok(())
}

/// Delete one native backend model directory and its Hugging Face lock entry.
fn remove_backend_model(path: &Path) -> anyhow::Result<()> {
    let existed = fs::symlink_metadata(path).is_ok();
    remove_tree(path)?;
    if existed {
        eprintln!("Removed cached model {}", path.display());
    }
    // Hugging Face stores per-model locks beside the hub directory.
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        let lock = parent.join(".locks").join(name);
        if fs::symlink_metadata(&lock).is_ok() {
            remove_tree(&lock)?;
            eprintln!("Removed cache lock {}", lock.display());
        }
    }
    Ok(())
}

/// Native model directories for the selected backends, optionally one model id.
fn native_model_paths(
    cache_root: &Path,
    modelscope_models: &Path,
    huggingface_hub: &Path,
    model_id: Option<&str>,
    targets: &BTreeSet<CacheSource>,
) -> anyhow::Result<Vec<PathBuf>> {
    let models = discover_models(cache_root, modelscope_models, huggingface_hub)?;
    let mut paths = Vec::new();
    for model in models {
        if model_id.is_some_and(|wanted| model.model_id != wanted) {
            continue;
        }
        for hit in model.hits {
            if hit.source != CacheSource::ModelHub && targets.contains(&hit.source) {
                paths.push(hit.path);
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

/// Remove models from modelhub and, when requested, from native backend caches.
///
/// `modelscope_models` is the `ModelScope` `models/` directory. `huggingface_hub`
/// is the Hugging Face hub directory. Dataset directories under those caches are
/// not scanned and are left in place.
fn clear(
    cache_root: &Path,
    modelscope_models: &Path,
    huggingface_hub: &Path,
    model_id: Option<&str>,
    all: bool,
    backend: Option<CacheSource>,
) -> anyhow::Result<()> {
    let targets = clear_targets(all, backend);
    let wipe_modelhub = targets.contains(&CacheSource::ModelHub);
    let wipe_native = targets
        .iter()
        .any(|source| *source != CacheSource::ModelHub);
    // `--all` without a model id removes every model. With an id, only that model.
    let wipe_everything = all && model_id.is_none();
    if wipe_everything && wipe_modelhub {
        refuse_unsafe_cache_root(cache_root)?;
    }
    // Snapshot native directories before the modelhub tree disappears.
    let native_paths = if wipe_native {
        native_model_paths(
            cache_root,
            modelscope_models,
            huggingface_hub,
            if wipe_everything { None } else { model_id },
            &targets,
        )?
    } else {
        Vec::new()
    };
    let mut removed_modelhub = false;
    if wipe_modelhub {
        let models = modelhub_models(cache_root)?;
        if wipe_everything {
            for model in &models {
                clear_model(cache_root, model)?;
            }
            if cache_root.exists() {
                fs::remove_dir_all(cache_root)
                    .with_context(|| format!("failed to clear {}", cache_root.display()))?;
            }
            removed_modelhub = true;
        } else if let Some(model) =
            model_id.and_then(|model_id| models.iter().find(|model| model.model_id == model_id))
        {
            clear_model(cache_root, model)?;
            removed_modelhub = true;
        }
    }
    if !wipe_everything && !removed_modelhub && native_paths.is_empty() {
        let model_id = model_id.context("model ID is required unless --all is used")?;
        let labels = targets
            .iter()
            .map(|source| source.label())
            .collect::<Vec<_>>()
            .join(", ");
        println!("Model `{model_id}` is not cached in {labels}");
        return Ok(());
    }
    for path in &native_paths {
        remove_backend_model(path)?;
    }
    let labels = targets
        .iter()
        .map(|source| source.label())
        .collect::<Vec<_>>()
        .join(", ");
    if wipe_everything {
        println!("Cleared models from {labels}");
    } else {
        let model_id = model_id.context("model ID is required unless --all is used")?;
        println!("Cleared `{model_id}` from {labels}");
    }
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
            backend,
            cache_dir,
        } => {
            let cache_dir = cache_dir.unwrap_or_else(modelhub::cache::cache_dir);
            clear(
                &cache_dir,
                &modelhub::modelscope::cache_dir().join("models"),
                &modelhub::huggingface::cache_dir(),
                model_id.as_deref(),
                all,
                backend,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::{CacheSource, clear, discover_models, model_disk_size, source_labels};
    use super::{Cli, ListRow, human_size, model_id_from_dir, render_model_table, shorten_home};
    use clap::Parser;
    #[cfg(unix)]
    use std::fs;
    use std::path::{Path, PathBuf};

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
    fn renders_list_as_a_bordered_table() {
        let rows = vec![
            ListRow {
                model_id: "acme/demo".to_owned(),
                size: "1.5 KiB".to_owned(),
                sources: "huggingface".to_owned(),
                paths: vec![PathBuf::from("/cache/hf")],
            },
            ListRow {
                model_id: "org/name".to_owned(),
                size: "2 B".to_owned(),
                sources: "modelscope".to_owned(),
                paths: vec![PathBuf::from("/cache/ms"), PathBuf::from("/cache/ms2")],
            },
        ];
        // A fixed width keeps the snapshot independent of the terminal size.
        let table = render_model_table(&rows, Some(64));
        assert_eq!(
            table,
            "\
╭───────────┬─────────┬─────────────┬────────────╮
│ MODEL     │    SIZE │ BACKENDS    │ PATH       │
╞═══════════╪═════════╪═════════════╪════════════╡
│ acme/demo │ 1.5 KiB │ huggingface │ /cache/hf  │
├───────────┼─────────┼─────────────┼────────────┤
│ org/name  │     2 B │ modelscope  │ /cache/ms  │
│           │         │             │ /cache/ms2 │
╰───────────┴─────────┴─────────────┴────────────╯"
        );
        assert_eq!(
            shorten_home("/Users/me/.cache/huggingface/hub", Some("/Users/me")),
            "~/.cache/huggingface/hub"
        );
    }

    #[test]
    fn clear_requires_a_model_or_all() {
        assert!(Cli::try_parse_from(["modelhub", "clear"]).is_err());
        assert!(Cli::try_parse_from(["modelhub", "clear", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "clear", "acme/demo", "--all"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "clear", "--all"]).is_ok());
        assert!(
            Cli::try_parse_from(["modelhub", "clear", "acme/demo", "--backend", "huggingface"])
                .is_ok()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "clear", "--all", "--backend", "modelscope"]).is_ok()
        );
        assert!(Cli::try_parse_from(["modelhub", "clear", "--backend", "huggingface"]).is_err());
        assert!(Cli::try_parse_from(["modelhub", "clear", "--backend", "nope"]).is_err());
    }

    #[test]
    fn download_jobs_must_be_positive() {
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "8"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "0"]).is_err());
    }

    /// Drop a temporary directory when the test finishes, including on failure.
    #[cfg(unix)]
    struct TempTree(PathBuf);

    #[cfg(unix)]
    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn discovers_every_backend_and_counts_shared_files_once() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "modelhub-discover-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let _cleanup = TempTree(root.clone());
        let modelhub_root = root.join("modelhub");
        let modelscope_models = root.join("modelscope");
        let huggingface_hub = root.join("huggingface");

        let owned = modelhub_root.join("models").join("acme--owned");
        fs::create_dir_all(&owned).unwrap();
        fs::write(owned.join(".modelhub-model-id"), "acme/owned").unwrap();
        fs::write(owned.join("weights.bin"), vec![0u8; 100]).unwrap();
        fs::create_dir_all(&huggingface_hub).unwrap();
        // The backend entry points at the modelhub copy and must not add a second size.
        symlink(&owned, huggingface_hub.join("models--acme--owned")).unwrap();

        let native_ms = modelscope_models.join("org--native");
        fs::create_dir_all(&native_ms).unwrap();
        fs::write(native_ms.join("config.json"), b"{}").unwrap();

        let hf_only = huggingface_hub.join("models--org--hfonly");
        let blob = hf_only.join("blobs").join("abc");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, vec![1u8; 50]).unwrap();
        let snapshot = hf_only.join("snapshots").join("main");
        fs::create_dir_all(&snapshot).unwrap();
        symlink("../../blobs/abc", snapshot.join("model.bin")).unwrap();

        let models = discover_models(&modelhub_root, &modelscope_models, &huggingface_hub).unwrap();
        assert_eq!(models.len(), 3);

        let owned = models
            .iter()
            .find(|model| model.model_id == "acme/owned")
            .unwrap();
        assert_eq!(source_labels(&owned.hits), "huggingface, modelhub");
        // `weights.bin` plus the model-id marker. The Hugging Face link is the same directory.
        assert_eq!(model_disk_size(&owned.hits).unwrap(), 110);

        let hf_only = models
            .iter()
            .find(|model| model.model_id == "org/hfonly")
            .unwrap();
        assert_eq!(source_labels(&hf_only.hits), "huggingface");
        // The snapshot symlink and the blob are the same inode.
        assert_eq!(model_disk_size(&hf_only.hits).unwrap(), 50);

        let native = models
            .iter()
            .find(|model| model.model_id == "org/native")
            .unwrap();
        assert_eq!(source_labels(&native.hits), "modelscope");
        assert_eq!(model_disk_size(&native.hits).unwrap(), 2);
    }

    /// A modelhub cache, one native model per backend, and dataset directories that must survive.
    #[cfg(unix)]
    fn write_clear_fixture(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let modelhub = root.join("modelhub");
        let modelscope_models = root.join("modelscope").join("models");
        let huggingface = root.join("huggingface");
        let owned = modelhub.join("models").join("acme--owned");
        fs::create_dir_all(&owned).unwrap();
        fs::write(owned.join("weights.bin"), b"hub").unwrap();
        let native = modelscope_models.join("org--native");
        fs::create_dir_all(&native).unwrap();
        fs::write(native.join("config.json"), b"{}").unwrap();
        let ms_dataset = root.join("modelscope").join("datasets").join("org--data");
        fs::create_dir_all(&ms_dataset).unwrap();
        fs::write(ms_dataset.join("data.txt"), b"keep").unwrap();
        let hf_model = huggingface.join("models--org--hf");
        fs::create_dir_all(&hf_model).unwrap();
        fs::write(hf_model.join("config.json"), b"{}").unwrap();
        let lock = huggingface.join(".locks").join("models--org--hf");
        fs::create_dir_all(&lock).unwrap();
        fs::write(lock.join("lock"), b"1").unwrap();
        fs::write(huggingface.join("CACHEDIR.TAG"), b"tag").unwrap();
        let hf_dataset = huggingface.join("datasets--org--data");
        fs::create_dir_all(&hf_dataset).unwrap();
        fs::write(hf_dataset.join("data.txt"), b"keep").unwrap();
        (modelhub, modelscope_models, huggingface)
    }

    #[cfg(unix)]
    fn temp_clear_root(name: &str) -> (TempTree, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "modelhub-clear-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        (TempTree(root.clone()), root)
    }

    #[cfg(unix)]
    #[test]
    fn clear_model_leaves_native_backend_caches() {
        let (_cleanup, root) = temp_clear_root("one");
        let (modelhub, modelscope_models, huggingface) = write_clear_fixture(&root);
        clear(
            &modelhub,
            &modelscope_models,
            &huggingface,
            Some("acme/owned"),
            false,
            None,
        )
        .unwrap();
        assert!(!modelhub.join("models").join("acme--owned").exists());
        assert!(modelscope_models.join("org--native").exists());
        assert!(huggingface.join("models--org--hf").exists());
    }

    #[cfg(unix)]
    #[test]
    fn clear_model_all_removes_that_model_from_every_backend() {
        let (_cleanup, root) = temp_clear_root("model-all");
        let (modelhub, modelscope_models, huggingface) = write_clear_fixture(&root);
        // The fixture's Hugging Face model is a different id and must stay.
        let linked = huggingface.join("models--acme--owned");
        std::os::unix::fs::symlink(modelhub.join("models").join("acme--owned"), &linked).unwrap();
        clear(
            &modelhub,
            &modelscope_models,
            &huggingface,
            Some("acme/owned"),
            true,
            None,
        )
        .unwrap();
        assert!(!modelhub.join("models").join("acme--owned").exists());
        assert!(!linked.exists());
        assert!(modelscope_models.join("org--native").exists());
        assert!(huggingface.join("models--org--hf").exists());
        assert!(modelhub.exists());
    }

    #[cfg(unix)]
    #[test]
    fn clear_all_removes_backend_models_and_keeps_datasets() {
        let (_cleanup, root) = temp_clear_root("all");
        let (modelhub, modelscope_models, huggingface) = write_clear_fixture(&root);
        clear(
            &modelhub,
            &modelscope_models,
            &huggingface,
            None,
            true,
            None,
        )
        .unwrap();
        assert!(!modelhub.exists());
        assert!(!modelscope_models.join("org--native").exists());
        assert!(!huggingface.join("models--org--hf").exists());
        assert!(!huggingface.join(".locks").join("models--org--hf").exists());
        assert!(
            root.join("modelscope")
                .join("datasets")
                .join("org--data")
                .join("data.txt")
                .exists()
        );
        assert!(
            huggingface
                .join("datasets--org--data")
                .join("data.txt")
                .exists()
        );
        assert!(huggingface.join("CACHEDIR.TAG").exists());
        assert!(huggingface.join(".locks").is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn clear_backend_removes_only_that_backend() {
        let (_cleanup, root) = temp_clear_root("backend");
        let (modelhub, modelscope_models, huggingface) = write_clear_fixture(&root);
        clear(
            &modelhub,
            &modelscope_models,
            &huggingface,
            None,
            true,
            Some(CacheSource::HuggingFace),
        )
        .unwrap();
        assert!(modelhub.join("models").join("acme--owned").exists());
        assert!(modelscope_models.join("org--native").exists());
        assert!(!huggingface.join("models--org--hf").exists());
        assert!(!huggingface.join(".locks").join("models--org--hf").exists());
        assert!(
            huggingface
                .join("datasets--org--data")
                .join("data.txt")
                .exists()
        );
    }
}
