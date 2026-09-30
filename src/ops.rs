//! High-level cache operations mirroring the CLI subcommands.
//!
//! Each public function returns structured data instead of printing, so it can
//! be called directly. The binary layer renders tables and summaries.

use crate::repos::{
    CacheSource, CachedRepo, MODEL_ID_FILE, RepoHit, RepoKind, RepoStatus, discover, disk_size,
    modelhub_cached, sources, status,
};
use crate::unified::{Backend, DownloadedRepo, RepoManifest};
use crate::upload::{
    UploadBackend, UploadOptions, UploadSummary, available_backends, collect_files,
};
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Options for [`download`].
#[derive(Clone, Debug)]
pub struct DownloadOptions {
    pub repo_id: String,
    /// Optional single file to download; `None` downloads the whole repository.
    ///
    /// A single file is fetched with one request per candidate (`kind` ×
    /// `backend`, narrowed by the hints) and never lists the repository.
    pub file: Option<String>,
    /// Restrict the download to a model or a dataset; `None` auto-detects it.
    pub kind: Option<RepoKind>,
    /// Restrict the download to one backend; `None` uses every supported backend.
    pub backend: Option<Backend>,
    /// Revision to request. Defaults to `main`/`master` per backend.
    pub revision: Option<String>,
    /// Root directory owned by modelhub.
    pub cache_root: PathBuf,
    /// Maximum number of files downloaded concurrently.
    pub jobs: usize,
    /// Keep both backend versions even when model weights differ.
    pub all_backends: bool,
    /// Show a progress bar.
    pub progress: bool,
}

impl DownloadOptions {
    /// Options for `repo_id`, defaulting to the modelhub cache and 4 jobs.
    #[must_use]
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            file: None,
            kind: None,
            backend: None,
            revision: None,
            cache_root: crate::cache::cache_dir(),
            jobs: 4,
            all_backends: false,
            progress: false,
        }
    }
}

/// Download a repository or a single file.
///
/// Auto-detects whether `repo_id` is a model or a dataset and which backends
/// host it unless [`DownloadOptions::kind`] or [`DownloadOptions::backend`] say
/// otherwise. A single-file download is stored in the modelhub cache only and
/// is not linked into the native backend caches.
pub async fn download(opts: &DownloadOptions) -> Result<DownloadedRepo> {
    fs::create_dir_all(&opts.cache_root)?;
    if let Some(file) = opts.file.as_deref() {
        return crate::unified::download_single_file(
            opts.kind,
            opts.backend,
            &opts.repo_id,
            opts.revision.as_deref(),
            file,
            &opts.cache_root,
            opts.progress,
        )
        .await;
    }
    let downloaded = crate::unified::download_repo(
        &opts.repo_id,
        opts.revision.as_deref().unwrap_or("main"),
        opts.revision.as_deref().unwrap_or("master"),
        &opts.cache_root,
        opts.jobs,
        opts.all_backends,
        opts.progress,
        opts.kind,
        opts.backend,
    )
    .await?;
    if let Some(root) = downloaded.huggingface_root.as_deref() {
        link_directory(
            root,
            &huggingface_cache_path(downloaded.kind, &opts.repo_id),
        )?;
    }
    if let Some(root) = downloaded.modelscope_root.as_deref() {
        link_directory(root, &modelscope_cache_path(downloaded.kind, &opts.repo_id))?;
    }
    Ok(downloaded)
}

/// Upload local files to Hugging Face and/or `ModelScope`.
///
/// Targets every backend with credentials unless `opts.backends` is set.
pub async fn upload(opts: &UploadOptions) -> Result<UploadSummary> {
    if opts.delete && opts.path_in_repo.is_none() && !crate::upload::has_directory(&opts.local) {
        bail!("`delete` needs a directory input or `path_in_repo` to scope deletions");
    }
    let files = collect_files(
        &opts.local,
        opts.path_in_repo.as_deref(),
        &opts.include,
        &opts.exclude,
    )?;
    let backends = if opts.backends.is_empty() {
        available_backends()
    } else {
        opts.backends.clone()
    };
    if backends.is_empty() {
        bail!("no upload credentials found; set HF_TOKEN or MODELSCOPE_API_TOKEN");
    }
    let mut results = Vec::new();
    for backend in backends {
        let result = match backend {
            UploadBackend::HuggingFace => crate::upload::huggingface::upload(opts, &files).await?,
            UploadBackend::ModelScope => crate::upload::modelscope::upload(opts, &files).await?,
        };
        results.push(result);
    }
    Ok(UploadSummary { results })
}

/// Options for [`list`].
#[derive(Clone, Debug)]
pub struct ListOptions {
    pub cache_root: PathBuf,
    /// Also verify each download and fill in [`RepoEntry::status`].
    pub check: bool,
    /// Override the `ModelScope` cache root (defaults to its own environment).
    pub modelscope_cache: Option<PathBuf>,
    /// Override the Hugging Face hub directory.
    pub huggingface_hub: Option<PathBuf>,
}
impl Default for ListOptions {
    fn default() -> Self {
        Self {
            cache_root: crate::cache::cache_dir(),
            check: false,
            modelscope_cache: None,
            huggingface_hub: None,
        }
    }
}

impl ListOptions {
    /// `ModelScope` cache directory [`list`] will scan.
    ///
    /// Uses [`Self::modelscope_cache`] when set. Otherwise follows
    /// `MODELSCOPE_CACHE`, then `~/.cache/modelscope`.
    #[must_use]
    pub fn modelscope_cache_dir(&self) -> PathBuf {
        self.modelscope_cache
            .clone()
            .unwrap_or_else(crate::modelscope::cache_dir)
    }

    /// Hugging Face hub directory [`list`] will scan.
    ///
    /// Uses [`Self::huggingface_hub`] when set. Otherwise follows
    /// `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then `~/.cache/huggingface/hub`.
    #[must_use]
    pub fn huggingface_cache_dir(&self) -> PathBuf {
        self.huggingface_hub
            .clone()
            .unwrap_or_else(crate::huggingface::cache_dir)
    }
}

/// One repository as presented to callers.
#[derive(Clone, Debug)]
pub struct RepoEntry {
    pub kind: RepoKind,
    pub id: String,
    /// Total bytes across every distinct copy on disk.
    pub size: u64,
    /// Offline completeness; `None` when not requested.
    pub status: Option<RepoStatus>,
    pub sources: Vec<CacheSource>,
    pub paths: Vec<PathBuf>,
    pub hits: Vec<RepoHit>,
}

/// List every cached repository visible to modelhub.
pub fn list(opts: &ListOptions) -> Result<Vec<RepoEntry>> {
    let (modelscope_cache, huggingface_hub) = resolve_backends(opts);
    let repos = discover(
        &opts.cache_root,
        modelscope_cache.as_path(),
        huggingface_hub.as_path(),
    )?;
    repos
        .iter()
        .map(|repo| {
            let status = opts.check.then(|| status(repo));
            entry(repo, status)
        })
        .collect()
}

/// Options for [`check`].
#[derive(Clone, Debug)]
pub struct CheckOptions {
    /// Only check this repository; `None` checks every cached repository.
    pub repo_id: Option<String>,
    pub cache_root: PathBuf,
    /// Never fetch remote manifests.
    pub offline: bool,
    pub modelscope_cache: Option<PathBuf>,
    pub huggingface_hub: Option<PathBuf>,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            repo_id: None,
            cache_root: crate::cache::cache_dir(),
            offline: false,
            modelscope_cache: None,
            huggingface_hub: None,
        }
    }
}

/// Verify repositories, fetching and recording manifests for native caches that
/// modelhub did not download itself.
pub async fn check(opts: &CheckOptions) -> Result<Vec<RepoEntry>> {
    let (modelscope_cache, huggingface_hub) = resolve_backends_check(opts);
    let mut repos = discover(&opts.cache_root, &modelscope_cache, &huggingface_hub)?;
    if let Some(repo_id) = opts.repo_id.as_deref() {
        repos.retain(|repo| repo.id == repo_id);
    }
    if !opts.offline {
        for repo in &repos {
            generate_manifests(repo).await;
        }
    }
    repos
        .iter()
        .map(|repo| entry(repo, Some(status(repo))))
        .collect()
}

/// Fetch and record a manifest for every hit that lacks one.
///
/// A repository with a modelhub-owned copy is fully described by that copy, so
/// its native links (which point into the modelhub tree) are skipped.
async fn generate_manifests(repo: &CachedRepo) {
    let owned: Vec<&RepoHit> = repo
        .hits
        .iter()
        .filter(|hit| hit.source == CacheSource::ModelHub)
        .collect();
    let targets: Vec<&RepoHit> = if owned.is_empty() {
        repo.hits
            .iter()
            .filter(|hit| {
                !fs::symlink_metadata(&hit.path).is_ok_and(|meta| meta.file_type().is_symlink())
            })
            .collect()
    } else {
        owned
    };
    for hit in targets {
        if hit.path.join(crate::unified::MANIFEST_FILE).is_file() {
            continue;
        }
        match build_manifest(hit, repo.kind, &repo.id).await {
            Ok(record) => {
                if let Err(error) = crate::unified::write_repo_manifest_record(&hit.path, &record) {
                    tracing::warn!("Failed to record manifest for {}: {error:#}", repo.id);
                }
            }
            Err(error) => tracing::warn!("Cannot build manifest for {}: {error:#}", repo.id),
        }
    }
}

/// Build a manifest for one repository directory by fetching remote file lists.
async fn build_manifest(hit: &RepoHit, kind: RepoKind, repo_id: &str) -> Result<RepoManifest> {
    let backends: &[&str] = match hit.source {
        CacheSource::ModelHub => &["huggingface", "modelscope"],
        CacheSource::HuggingFace => &["huggingface"],
        CacheSource::ModelScope => &["modelscope"],
    };
    let mut entries = BTreeMap::new();
    for backend in backends {
        // modelhub's content-addressed cache nests `<backend>/snapshots`; older
        // layouts and native caches keep `snapshots` directly.
        let nested = hit.path.join(backend).join("snapshots");
        let base = if nested.is_dir() {
            nested
        } else {
            hit.path.join("snapshots")
        };
        let Some(revision) = first_revision(&base) else {
            continue;
        };
        match crate::unified::fetch_backend_manifest(backend, kind, repo_id, &revision).await {
            Ok(entry) => {
                entries.insert((*backend).to_owned(), entry);
            }
            Err(error) => {
                tracing::warn!("Cannot fetch {backend} manifest for {repo_id}: {error:#}");
            }
        }
    }
    if entries.is_empty() {
        bail!("no remote manifest could be fetched");
    }
    Ok(RepoManifest {
        version: 1,
        kind,
        backends: entries,
    })
}

/// Name of the first snapshot revision directory under `base`, if any.
fn first_revision(base: &Path) -> Option<String> {
    let mut revisions: Vec<String> = fs::read_dir(base)
        .ok()?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .collect();
    revisions.sort();
    revisions.into_iter().next()
}

/// Options for [`clear`].
#[derive(Clone, Debug)]
pub struct ClearOptions {
    /// Repository to remove. Required unless `all` is set.
    pub repo_id: Option<String>,
    /// Remove every repository. With a `repo_id`, remove just that one.
    pub all: bool,
    /// Restrict to one cache; otherwise `all` covers every supported cache.
    pub backend: Option<CacheSource>,
    pub cache_root: PathBuf,
    pub modelscope_cache: Option<PathBuf>,
    pub huggingface_hub: Option<PathBuf>,
}

impl Default for ClearOptions {
    fn default() -> Self {
        Self {
            repo_id: None,
            all: false,
            backend: None,
            cache_root: crate::cache::cache_dir(),
            modelscope_cache: None,
            huggingface_hub: None,
        }
    }
}

/// What [`clear`] removed.
#[derive(Clone, Debug)]
pub struct ClearSummary {
    /// Backends the request applied to.
    pub targets: Vec<CacheSource>,
    /// Paths removed, in removal order.
    pub removed: Vec<PathBuf>,
    /// Whether a matching modelhub or native copy existed.
    pub found: bool,
}

/// Remove repositories from modelhub and, when requested, from native caches.
pub fn clear(opts: &ClearOptions) -> Result<ClearSummary> {
    let (modelscope_cache, huggingface_hub) = resolve_backends_clear(opts);
    let repo_id = opts.repo_id.as_deref();
    if repo_id.is_none() && !opts.all {
        bail!("repository ID is required unless `all` is set");
    }
    let targets = clear_targets(opts.all, opts.backend);
    let wipe_modelhub = targets.contains(&CacheSource::ModelHub);
    let wipe_native = targets
        .iter()
        .any(|source| *source != CacheSource::ModelHub);
    // `all` without an id removes every repository. With an id, only that id.
    let wipe_everything = opts.all && repo_id.is_none();
    if wipe_everything && wipe_modelhub {
        refuse_unsafe_cache_root(&opts.cache_root)?;
    }
    // Snapshot native directories before the modelhub tree disappears.
    let native_paths = if wipe_native {
        native_repo_paths(
            &opts.cache_root,
            &modelscope_cache,
            &huggingface_hub,
            if wipe_everything { None } else { repo_id },
            &targets,
        )?
    } else {
        Vec::new()
    };
    let mut removed = Vec::new();
    let mut removed_modelhub = false;
    if wipe_modelhub {
        let repos = modelhub_cached(&opts.cache_root)?;
        if wipe_everything {
            for repo in &repos {
                clear_repo(&opts.cache_root, repo, &mut removed)?;
            }
            if opts.cache_root.exists() {
                fs::remove_dir_all(&opts.cache_root)
                    .with_context(|| format!("failed to clear {}", opts.cache_root.display()))?;
            }
            removed_modelhub = true;
        } else if let Some(repo_id) = repo_id {
            // The same identifier may exist as both a model and a dataset.
            for repo in repos.iter().filter(|repo| repo.id == repo_id) {
                clear_repo(&opts.cache_root, repo, &mut removed)?;
                removed_modelhub = true;
            }
        }
    }
    for path in &native_paths {
        remove_backend_repo(path, &mut removed)?;
    }
    let found =
        wipe_everything || removed_modelhub || !native_paths.is_empty() || !removed.is_empty();
    Ok(ClearSummary {
        targets: targets.into_iter().collect(),
        removed,
        found,
    })
}

fn resolve_backends(opts: &ListOptions) -> (PathBuf, PathBuf) {
    (opts.modelscope_cache_dir(), opts.huggingface_cache_dir())
}

fn resolve_backends_check(opts: &CheckOptions) -> (PathBuf, PathBuf) {
    (
        opts.modelscope_cache
            .clone()
            .unwrap_or_else(crate::modelscope::cache_dir),
        opts.huggingface_hub
            .clone()
            .unwrap_or_else(crate::huggingface::cache_dir),
    )
}

fn resolve_backends_clear(opts: &ClearOptions) -> (PathBuf, PathBuf) {
    (
        opts.modelscope_cache
            .clone()
            .unwrap_or_else(crate::modelscope::cache_dir),
        opts.huggingface_hub
            .clone()
            .unwrap_or_else(crate::huggingface::cache_dir),
    )
}

fn entry(repo: &CachedRepo, status: Option<RepoStatus>) -> Result<RepoEntry> {
    Ok(RepoEntry {
        kind: repo.kind,
        id: repo.id.clone(),
        size: disk_size(&repo.hits)?,
        status,
        sources: sources(&repo.hits),
        paths: crate::repos::display_paths(&repo.hits),
        hits: repo.hits.clone(),
    })
}

fn link_directory(source: &Path, target: &Path) -> Result<()> {
    let source = fs::canonicalize(source)
        .with_context(|| format!("modelhub cache path does not exist: {}", source.display()))?;
    if fs::canonicalize(target).is_ok_and(|existing| existing == source) {
        tracing::warn!("cache link already exists: {}", target.display());
        return Ok(());
    }
    if let Ok(metadata) = fs::symlink_metadata(target) {
        if metadata.file_type().is_symlink() {
            tracing::warn!("preserving existing backend link: {}", target.display());
            return Ok(());
        } else if metadata.is_dir() {
            if fs::read_dir(target)?.next().is_none() {
                // An empty backend-created directory can safely be replaced by the link.
                fs::remove_dir(target)?;
            } else {
                // Preserve an existing native cache and add links for entries it does not have.
                tracing::warn!(
                    "backend cache already exists; preserving and reusing it: {}",
                    target.display()
                );
                merge_directory(&source, target)?;
                return Ok(());
            }
        } else if fs::canonicalize(target).is_ok_and(|existing| existing == source) {
            return Ok(());
        } else {
            tracing::warn!(
                "preserving existing backend cache path: {}",
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

fn merge_directory(source: &Path, target: &Path) -> Result<()> {
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

fn backend_cache_paths(kind: RepoKind, repo_id: &str) -> [PathBuf; 2] {
    [
        modelscope_cache_path(kind, repo_id),
        huggingface_cache_path(kind, repo_id),
    ]
}

fn modelscope_cache_path(kind: RepoKind, repo_id: &str) -> PathBuf {
    crate::modelscope::cache_dir()
        .join(kind.segment())
        .join(repo_id.replace('/', "--"))
}

fn huggingface_cache_path(kind: RepoKind, repo_id: &str) -> PathBuf {
    crate::huggingface::cache_dir().join(format!(
        "{}{}",
        hf_prefix(kind),
        repo_id.replace('/', "--")
    ))
}

/// Directory name prefix used by the Hugging Face hub for each kind.
const fn hf_prefix(kind: RepoKind) -> &'static str {
    match kind {
        RepoKind::Model => "models--",
        RepoKind::Dataset => "datasets--",
    }
}

fn remove_links_into(
    path: &Path,
    cache_root: &Path,
    repo_sources: &[PathBuf],
    removed: &mut Vec<PathBuf>,
) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.file_type().is_symlink() {
        if fs::canonicalize(path)
            .is_ok_and(|target| target.starts_with(cache_root) || repo_sources.contains(&target))
        {
            fs::remove_file(path)?;
            removed.push(path.to_path_buf());
        }
        return Ok(());
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            remove_links_into(&entry?.path(), cache_root, repo_sources, removed)?;
        }
    }
    Ok(())
}

fn clear_repo(cache_root: &Path, repo: &CachedRepo, removed: &mut Vec<PathBuf>) -> Result<()> {
    let canonical_root = fs::canonicalize(cache_root).unwrap_or_else(|_| cache_root.to_path_buf());
    let repo_sources: Vec<_> = repo
        .hits
        .iter()
        .filter_map(|hit| fs::canonicalize(&hit.path).ok())
        .collect();
    for backend_path in backend_cache_paths(repo.kind, &repo.id) {
        remove_links_into(&backend_path, &canonical_root, &repo_sources, removed)?;
    }
    for hit in &repo.hits {
        let metadata = fs::symlink_metadata(&hit.path)?;
        if metadata.file_type().is_symlink() {
            fs::remove_file(&hit.path)?;
        } else {
            fs::remove_dir_all(&hit.path)?;
        }
        removed.push(hit.path.clone());
    }
    garbage_collect_blobs(cache_root)?;
    Ok(())
}

fn garbage_collect_blobs(cache_root: &Path) -> Result<()> {
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

/// Backends selected by `all` and `backend`.
///
/// A named backend limits the command to that cache. `all` without `backend`
/// covers every supported cache. An id without `all` or `backend` stays on
/// modelhub.
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
fn refuse_unsafe_cache_root(cache_root: &Path) -> Result<()> {
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
fn refuse_unsafe_removal(path: &Path) -> Result<()> {
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
fn remove_tree(path: &Path) -> Result<()> {
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

/// Delete one native backend repository directory and its Hugging Face lock entry.
fn remove_backend_repo(path: &Path, removed: &mut Vec<PathBuf>) -> Result<()> {
    let existed = fs::symlink_metadata(path).is_ok();
    remove_tree(path)?;
    if existed {
        removed.push(path.to_path_buf());
    }
    // Hugging Face stores per-repository locks beside the hub directory.
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        let lock = parent.join(".locks").join(name);
        if fs::symlink_metadata(&lock).is_ok() {
            remove_tree(&lock)?;
            removed.push(lock);
        }
    }
    Ok(())
}

/// Native repository directories for the selected backends, optionally one id.
fn native_repo_paths(
    cache_root: &Path,
    modelscope_cache: &Path,
    huggingface_hub: &Path,
    repo_id: Option<&str>,
    targets: &BTreeSet<CacheSource>,
) -> Result<Vec<PathBuf>> {
    let repos = discover(cache_root, modelscope_cache, huggingface_hub)?;
    let mut paths = Vec::new();
    for repo in repos {
        if repo_id.is_some_and(|wanted| repo.id != wanted) {
            continue;
        }
        for hit in repo.hits {
            if hit.source != CacheSource::ModelHub && targets.contains(&hit.source) {
                paths.push(hit.path);
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::{CacheSource, ClearOptions, clear};
    use std::fs;
    use std::path::{Path, PathBuf};

    struct TempTree(PathBuf);

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root(name: &str) -> (TempTree, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "modelhub-ops-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        (TempTree(root.clone()), root)
    }

    /// A modelhub cache, native repos per backend, and dataset directories.
    fn write_clear_fixture(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let modelhub = root.join("modelhub");
        let modelscope_cache = root.join("modelscope");
        let huggingface = root.join("huggingface");
        let owned = modelhub.join("models").join("acme--owned");
        fs::create_dir_all(&owned).unwrap();
        fs::write(owned.join("weights.bin"), b"hub").unwrap();
        let native = modelscope_cache.join("models").join("org--native");
        fs::create_dir_all(&native).unwrap();
        fs::write(native.join("config.json"), b"{}").unwrap();
        let ms_dataset = modelscope_cache.join("datasets").join("org--data");
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
        (modelhub, modelscope_cache, huggingface)
    }

    fn options(
        root: &Path,
        modelhub: &Path,
        modelscope: &Path,
        huggingface: &Path,
        repo_id: Option<&str>,
        all: bool,
        backend: Option<CacheSource>,
    ) -> ClearOptions {
        let _ = root;
        ClearOptions {
            repo_id: repo_id.map(str::to_owned),
            all,
            backend,
            cache_root: modelhub.to_path_buf(),
            modelscope_cache: Some(modelscope.to_path_buf()),
            huggingface_hub: Some(huggingface.to_path_buf()),
        }
    }

    #[cfg(unix)]
    #[test]
    fn clear_repo_leaves_native_backend_caches() {
        let (_cleanup, root) = temp_root("one");
        let (modelhub, modelscope, huggingface) = write_clear_fixture(&root);
        clear(&options(
            &root,
            &modelhub,
            &modelscope,
            &huggingface,
            Some("acme/owned"),
            false,
            None,
        ))
        .unwrap();
        assert!(!modelhub.join("models").join("acme--owned").exists());
        assert!(modelscope.join("models").join("org--native").exists());
        assert!(huggingface.join("models--org--hf").exists());
    }

    #[cfg(unix)]
    #[test]
    fn clear_all_backend_removes_every_backend() {
        let (_cleanup, root) = temp_root("model-all");
        let (modelhub, modelscope, huggingface) = write_clear_fixture(&root);
        let linked = huggingface.join("models--acme--owned");
        std::os::unix::fs::symlink(modelhub.join("models").join("acme--owned"), &linked).unwrap();
        clear(&options(
            &root,
            &modelhub,
            &modelscope,
            &huggingface,
            Some("acme/owned"),
            true,
            None,
        ))
        .unwrap();
        assert!(!modelhub.join("models").join("acme--owned").exists());
        assert!(!linked.exists());
        assert!(modelscope.join("models").join("org--native").exists());
        assert!(huggingface.join("models--org--hf").exists());
        assert!(modelhub.exists());
    }

    #[cfg(unix)]
    #[test]
    fn clear_all_removes_backend_models_and_datasets() {
        let (_cleanup, root) = temp_root("all");
        let (modelhub, modelscope, huggingface) = write_clear_fixture(&root);
        clear(&options(
            &root,
            &modelhub,
            &modelscope,
            &huggingface,
            None,
            true,
            None,
        ))
        .unwrap();
        assert!(!modelhub.exists());
        assert!(!modelscope.join("models").join("org--native").exists());
        assert!(!huggingface.join("models--org--hf").exists());
        assert!(!huggingface.join(".locks").join("models--org--hf").exists());
        assert!(!modelscope.join("datasets").join("org--data").exists());
        assert!(!huggingface.join("datasets--org--data").exists());
        assert!(huggingface.join("CACHEDIR.TAG").exists());
    }

    #[cfg(unix)]
    #[test]
    fn clear_backend_removes_only_that_backend() {
        let (_cleanup, root) = temp_root("backend");
        let (modelhub, modelscope, huggingface) = write_clear_fixture(&root);
        clear(&options(
            &root,
            &modelhub,
            &modelscope,
            &huggingface,
            None,
            true,
            Some(CacheSource::HuggingFace),
        ))
        .unwrap();
        assert!(modelhub.join("models").join("acme--owned").exists());
        assert!(modelscope.join("models").join("org--native").exists());
        assert!(!huggingface.join("models--org--hf").exists());
        assert!(!huggingface.join("datasets--org--data").exists());
        assert!(modelscope.join("datasets").join("org--data").exists());
    }

    #[test]
    fn clear_requires_an_id_or_all() {
        let opts = ClearOptions {
            repo_id: None,
            all: false,
            ..ClearOptions::default()
        };
        assert!(clear(&opts).is_err());
    }
}
