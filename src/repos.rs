//! Discover models and datasets across the modelhub, `ModelScope`, and Hugging
//! Face caches, and verify them against recorded manifests.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Marker file recording the repository ID inside a modelhub-owned directory.
pub(crate) const MODEL_ID_FILE: &str = ".modelhub-model-id";

/// Whether a repository holds a model or a dataset.
///
/// Variant order places models before datasets, so listings group predictably.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoKind {
    Model,
    Dataset,
}

impl RepoKind {
    /// URL segment and cache directory name for this kind (`models`/`datasets`).
    #[must_use]
    pub const fn segment(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }

    /// Singular label used in user-facing messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Dataset => "dataset",
        }
    }
}

/// Cache that contributed a discovered repository directory.
///
/// Variant order is alphabetical so a `BTreeSet` prints stable backend labels.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CacheSource {
    HuggingFace,
    ModelHub,
    ModelScope,
}

impl CacheSource {
    /// Column label for this cache.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelHub => "modelhub",
            Self::ModelScope => "modelscope",
        }
    }
}

/// One on-disk directory that belongs to a repository.
#[derive(Clone, Debug)]
pub struct RepoHit {
    pub source: CacheSource,
    pub path: PathBuf,
}

/// A model or dataset found in one or more caches.
#[derive(Clone, Debug)]
pub struct CachedRepo {
    pub kind: RepoKind,
    pub id: String,
    pub hits: Vec<RepoHit>,
}

/// Offline completeness of a repository, judged from its recorded manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepoStatus {
    Complete,
    Incomplete { present: usize, total: usize },
    Unknown,
}

impl fmt::Display for RepoStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complete => formatter.write_str("complete"),
            Self::Incomplete { present, total } => {
                write!(formatter, "incomplete {present}/{total}")
            }
            Self::Unknown => formatter.write_str("unknown"),
        }
    }
}

/// Repositories discovered while scanning caches, keyed by kind and identifier.
type RepoMap = BTreeMap<(RepoKind, String), Vec<RepoHit>>;

/// Decode a repository directory into its kind and `org/name` identifier.
///
/// `kind` is the kind implied by the parent directory. For the Hugging Face hub
/// layout, the `models--`/`datasets--` prefix overrides it. A
/// `.modelhub-model-id` marker, when present, supplies the exact identifier.
fn repo_from_dir(
    path: &Path,
    kind: RepoKind,
    huggingface_layout: bool,
) -> Option<(RepoKind, String)> {
    let mut kind = kind;
    let mut name = path.file_name()?.to_str()?;
    if huggingface_layout {
        if let Some(rest) = name.strip_prefix("models--") {
            kind = RepoKind::Model;
            name = rest;
        } else {
            let rest = name.strip_prefix("datasets--")?;
            kind = RepoKind::Dataset;
            name = rest;
        }
    }
    let marker = path.join(MODEL_ID_FILE);
    if let Ok(repo_id) = fs::read_to_string(marker) {
        let repo_id = repo_id.trim();
        if !repo_id.is_empty() {
            return Some((kind, repo_id.to_owned()));
        }
    }
    let (namespace, repo) = name.split_once("--")?;
    Some((kind, format!("{namespace}/{repo}")))
}

/// Record repository directories directly under `parent`.
///
/// `kind` is the kind implied by `parent`; `huggingface_layout` decodes the
/// `models--org--name` / `datasets--org--name` names used by the Hugging Face
/// hub. Other caches use `org--name`. A symlink to a directory is included, so a
/// backend link into the modelhub cache still shows up under that backend.
fn collect_cache_parent(
    repos: &mut RepoMap,
    parent: &Path,
    kind: RepoKind,
    huggingface_layout: bool,
    source: CacheSource,
) -> Result<()> {
    if !parent.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(parent)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(key) = repo_from_dir(&path, kind, huggingface_layout) {
            remember(repos, key, RepoHit { source, path });
        }
    }
    Ok(())
}

/// Keep the first hit when the same source already recorded this directory.
fn remember(repos: &mut RepoMap, key: (RepoKind, String), hit: RepoHit) {
    let hits = repos.entry(key).or_default();
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
fn collect_modelhub(repos: &mut RepoMap, cache_root: &Path) -> Result<()> {
    collect_cache_parent(
        repos,
        &cache_root.join("models"),
        RepoKind::Model,
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        repos,
        &cache_root.join("datasets"),
        RepoKind::Dataset,
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        repos,
        &cache_root.join("modelscope").join("models"),
        RepoKind::Model,
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        repos,
        &cache_root.join("modelscope").join("datasets"),
        RepoKind::Dataset,
        false,
        CacheSource::ModelHub,
    )?;
    collect_cache_parent(
        repos,
        &cache_root.join("huggingface").join("hub"),
        RepoKind::Model,
        true,
        CacheSource::ModelHub,
    )?;
    Ok(())
}

/// Models and datasets stored in the modelhub cache. Native backend copies are
/// excluded so callers can operate only on modelhub-owned directories.
#[must_use = "returns the discovered repositories"]
pub fn modelhub_cached(cache_root: &Path) -> Result<Vec<CachedRepo>> {
    let mut repos = RepoMap::new();
    collect_modelhub(&mut repos, cache_root)?;
    Ok(to_cached(repos))
}

/// Models and datasets visible to modelhub, `ModelScope`, and Hugging Face.
///
/// `modelscope_cache` is the `ModelScope` cache root containing `models/` and
/// `datasets/`. `huggingface_hub` is the Hugging Face hub directory containing
/// `models--*` and `datasets--*` folders.
pub fn discover(
    modelhub_root: &Path,
    modelscope_cache: &Path,
    huggingface_hub: &Path,
) -> Result<Vec<CachedRepo>> {
    let mut repos = RepoMap::new();
    collect_modelhub(&mut repos, modelhub_root)?;
    collect_cache_parent(
        &mut repos,
        &modelscope_cache.join("models"),
        RepoKind::Model,
        false,
        CacheSource::ModelScope,
    )?;
    collect_cache_parent(
        &mut repos,
        &modelscope_cache.join("datasets"),
        RepoKind::Dataset,
        false,
        CacheSource::ModelScope,
    )?;
    collect_cache_parent(
        &mut repos,
        huggingface_hub,
        RepoKind::Model,
        true,
        CacheSource::HuggingFace,
    )?;
    Ok(to_cached(repos))
}

fn to_cached(repos: RepoMap) -> Vec<CachedRepo> {
    repos
        .into_iter()
        .map(|((kind, id), hits)| CachedRepo { kind, id, hits })
        .collect()
}

/// Canonical roots, omitting a path that already lives inside another hit.
///
/// modelhub links a backend cache entry at the snapshot directory, which sits
/// inside the modelhub repository directory. Counting both would report the same
/// files twice.
fn outermost_paths(hits: &[RepoHit]) -> Vec<PathBuf> {
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
fn representative_path(hits: &[RepoHit], root: &Path) -> PathBuf {
    let mut matches: Vec<&RepoHit> = hits
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

/// Paths to report for a repository: one per distinct copy on disk.
#[must_use]
pub fn display_paths(hits: &[RepoHit]) -> Vec<PathBuf> {
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
) -> Result<u64> {
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
) -> Result<u64> {
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
/// beside the repository directory; directory targets must stay inside `root`.
fn linked_entry_size(
    root: &Path,
    path: &Path,
    files: &mut HashSet<(u64, u64)>,
    dirs: &mut HashSet<PathBuf>,
) -> Result<u64> {
    match fs::metadata(path) {
        Ok(target) if target.is_file() => Ok(account_file(&target, files)),
        Ok(target) if target.is_dir() => directory_size_within(root, path, files, dirs),
        _ => Ok(0),
    }
}

/// Disk usage of every distinct copy of a repository.
pub fn disk_size(hits: &[RepoHit]) -> Result<u64> {
    let mut files = HashSet::new();
    let mut dirs = HashSet::new();
    let mut total = 0;
    for root in outermost_paths(hits) {
        total += directory_size(&root, &mut files, &mut dirs)?;
    }
    Ok(total)
}

/// Distinct backends that hold this repository, in stable order.
#[must_use]
pub fn sources(hits: &[RepoHit]) -> Vec<CacheSource> {
    hits.iter()
        .map(|hit| hit.source)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Offline completeness, aggregated across every hit that carries a manifest.
///
/// Hits without a manifest are ignored; with no manifest anywhere this is
/// [`RepoStatus::Unknown`].
#[must_use]
pub fn status(repo: &CachedRepo) -> RepoStatus {
    let mut present = 0usize;
    let mut total = 0usize;
    let mut verified = false;
    for hit in &repo.hits {
        if !hit.path.join(crate::unified::MANIFEST_FILE).is_file() {
            continue;
        }
        verified = true;
        let (hit_present, hit_total) = repo_counts(&hit.path);
        present += hit_present;
        total += hit_total;
    }
    if !verified {
        RepoStatus::Unknown
    } else if present == total {
        RepoStatus::Complete
    } else {
        RepoStatus::Incomplete { present, total }
    }
}

/// Count expected and present files for one manifest-bearing directory.
fn repo_counts(repo_root: &Path) -> (usize, usize) {
    let Some(manifest) = crate::unified::read_repo_manifest(repo_root) else {
        return (0, 0);
    };
    let mut present = 0usize;
    let mut total = 0usize;
    for (backend, entry) in &manifest.backends {
        for (path, size) in &entry.files {
            total += 1;
            if snapshot_has(repo_root, backend, &entry.revision, path, *size) {
                present += 1;
            }
        }
    }
    (present, total)
}

/// Whether a snapshot holds `path` at `size` bytes.
///
/// Handles both layouts: `snapshots/<revision>` (native backend caches) and
/// `<backend>/snapshots/<revision>` (modelhub's content-addressed cache).
/// Absolute and parent-directory paths are rejected so a manifest cannot escape
/// the snapshot root.
fn snapshot_has(repo_root: &Path, backend: &str, revision: &str, path: &str, size: u64) -> bool {
    let relative = Path::new(path);
    if relative.is_absolute()
        || relative.components().any(|part| {
            !matches!(
                part,
                std::path::Component::Normal(_) | std::path::Component::CurDir
            )
        })
    {
        return false;
    }
    [
        repo_root.join("snapshots").join(revision),
        repo_root.join(backend).join("snapshots").join(revision),
    ]
    .iter()
    .any(|snapshot| {
        fs::metadata(snapshot.join(relative))
            .is_ok_and(|metadata| metadata.is_file() && (size == 0 || metadata.len() == size))
    })
}

#[cfg(test)]
mod tests {
    use super::{CacheSource, RepoKind, RepoStatus, discover, disk_size, status};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Drop a temporary directory when the test finishes, including on failure.
    struct TempTree(PathBuf);

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root(name: &str) -> (TempTree, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "modelhub-repos-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        (TempTree(root.clone()), root)
    }

    #[test]
    fn decodes_cache_directory_names() {
        assert_eq!(
            super::repo_from_dir(Path::new("/cache/acme--demo--v2"), RepoKind::Model, false),
            Some((RepoKind::Model, "acme/demo--v2".to_owned()))
        );
        assert_eq!(
            super::repo_from_dir(
                Path::new("/cache/models--acme--demo"),
                RepoKind::Model,
                true
            ),
            Some((RepoKind::Model, "acme/demo".to_owned()))
        );
        assert_eq!(
            super::repo_from_dir(
                Path::new("/cache/datasets--acme--demo"),
                RepoKind::Model,
                true
            ),
            Some((RepoKind::Dataset, "acme/demo".to_owned()))
        );
    }

    #[cfg(unix)]
    #[test]
    fn discovers_every_backend_and_counts_shared_files_once() {
        use std::os::unix::fs::symlink;

        let (_cleanup, root) = temp_root("discover");
        let modelhub_root = root.join("modelhub");
        let modelscope_cache = root.join("modelscope");
        let huggingface_hub = root.join("huggingface");

        let owned = modelhub_root.join("models").join("acme--owned");
        fs::create_dir_all(&owned).unwrap();
        fs::write(owned.join(".modelhub-model-id"), "acme/owned").unwrap();
        fs::write(owned.join("weights.bin"), vec![0u8; 100]).unwrap();
        fs::create_dir_all(&huggingface_hub).unwrap();
        symlink(&owned, huggingface_hub.join("models--acme--owned")).unwrap();

        let native_ms = modelscope_cache.join("models").join("org--native");
        fs::create_dir_all(&native_ms).unwrap();
        fs::write(native_ms.join("config.json"), b"{}").unwrap();

        let hf_only = huggingface_hub.join("models--org--hfonly");
        let blob = hf_only.join("blobs").join("abc");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, vec![1u8; 50]).unwrap();
        let snapshot = hf_only.join("snapshots").join("main");
        fs::create_dir_all(&snapshot).unwrap();
        symlink("../../blobs/abc", snapshot.join("model.bin")).unwrap();

        let ms_dataset = modelscope_cache.join("datasets").join("org--data");
        fs::create_dir_all(&ms_dataset).unwrap();
        fs::write(ms_dataset.join("data.txt"), b"set").unwrap();
        let hf_dataset = huggingface_hub.join("datasets--org--hfdata");
        fs::create_dir_all(&hf_dataset).unwrap();
        fs::write(hf_dataset.join("data.txt"), b"set").unwrap();

        let repos = discover(&modelhub_root, &modelscope_cache, &huggingface_hub).unwrap();
        assert_eq!(repos.len(), 5);

        let owned = find(&repos, "acme/owned");
        assert_eq!(owned.kind, RepoKind::Model);
        assert_eq!(
            super::sources(&owned.hits),
            vec![CacheSource::HuggingFace, CacheSource::ModelHub]
        );
        assert_eq!(disk_size(&owned.hits).unwrap(), 110);

        let hf_only = find(&repos, "org/hfonly");
        assert_eq!(
            super::sources(&hf_only.hits),
            vec![CacheSource::HuggingFace]
        );
        assert_eq!(disk_size(&hf_only.hits).unwrap(), 50);

        let native = find(&repos, "org/native");
        assert_eq!(super::sources(&native.hits), vec![CacheSource::ModelScope]);
        assert_eq!(disk_size(&native.hits).unwrap(), 2);

        let ms_dataset = find(&repos, "org/data");
        assert_eq!(ms_dataset.kind, RepoKind::Dataset);
        assert_eq!(
            super::sources(&ms_dataset.hits),
            vec![CacheSource::ModelScope]
        );

        let hf_dataset = find(&repos, "org/hfdata");
        assert_eq!(hf_dataset.kind, RepoKind::Dataset);
        assert_eq!(
            super::sources(&hf_dataset.hits),
            vec![CacheSource::HuggingFace]
        );
    }

    #[cfg(unix)]
    fn find<'a>(repos: &'a [super::CachedRepo], id: &str) -> &'a super::CachedRepo {
        repos.iter().find(|repo| repo.id == id).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn reports_completion_from_the_manifest() {
        let (_cleanup, root) = temp_root("status");
        fs::create_dir_all(&root).unwrap();
        let manifest = r#"{"version":1,"kind":"model","backends":{"huggingface":{"revision":"abc","files":{"config.json":2,"weights.bin":4}}}}"#;
        fs::write(root.join(crate::unified::MANIFEST_FILE), manifest).unwrap();
        let snapshot = root.join("huggingface").join("snapshots").join("abc");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        fs::write(snapshot.join("weights.bin"), b"1234").unwrap();
        assert_eq!(status(&cached(&root)), RepoStatus::Complete);

        fs::write(snapshot.join("weights.bin"), b"12").unwrap();
        assert_eq!(
            status(&cached(&root)),
            RepoStatus::Incomplete {
                present: 1,
                total: 2
            }
        );

        fs::remove_file(root.join(crate::unified::MANIFEST_FILE)).unwrap();
        assert_eq!(status(&cached(&root)), RepoStatus::Unknown);
    }

    #[cfg(unix)]
    #[test]
    fn verifies_native_snapshot_layout() {
        let (_cleanup, root) = temp_root("native");
        let snapshot = root.join("snapshots").join("main");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("data.txt"), b"ok").unwrap();
        let manifest = r#"{"version":1,"kind":"dataset","backends":{"modelscope":{"revision":"main","files":{"data.txt":2}}}}"#;
        fs::write(root.join(crate::unified::MANIFEST_FILE), manifest).unwrap();
        assert_eq!(status(&cached(&root)), RepoStatus::Complete);

        fs::remove_file(snapshot.join("data.txt")).unwrap();
        assert_eq!(
            status(&cached(&root)),
            RepoStatus::Incomplete {
                present: 0,
                total: 1
            }
        );
    }

    #[cfg(unix)]
    fn cached(root: &Path) -> super::CachedRepo {
        super::CachedRepo {
            kind: RepoKind::Model,
            id: "acme/demo".to_owned(),
            hits: vec![super::RepoHit {
                source: CacheSource::ModelHub,
                path: root.to_path_buf(),
            }],
        }
    }
}
