//! Upload a local file or directory to Hugging Face or `ModelScope`.
//!
//! The public entry point is [`crate::ops::upload`]; this module holds the
//! shared option/result types plus the two backend implementations.

pub mod huggingface;
pub mod modelscope;

use crate::repos::RepoKind;
use crate::unified::RemoteEntry;
use anyhow::{Context, Result, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// A backend an upload can target.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum UploadBackend {
    HuggingFace,
    ModelScope,
}

impl UploadBackend {
    /// Label used in messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }
}

/// Options for [`crate::ops::upload`].
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug)]
pub struct UploadOptions {
    /// Repository identifier, for example `org/name`.
    pub repo_id: String,
    /// Whether the remote repository is a model or a dataset.
    pub kind: RepoKind,
    /// Local files or directories to upload.
    pub local: Vec<PathBuf>,
    /// Optional sub-path inside the repository for the uploaded content.
    pub path_in_repo: Option<String>,
    /// Target branch/revision. Defaults to `main`/`master` per backend.
    pub revision: Option<String>,
    /// Commit message.
    pub commit_message: Option<String>,
    /// Create the repository when it does not exist.
    pub create: bool,
    /// Create the repository as private.
    pub private: bool,
    /// Backends to upload to. Empty means every backend with credentials.
    pub backends: Vec<UploadBackend>,
    /// Glob patterns; when set, only matching files are uploaded.
    pub include: Vec<String>,
    /// Glob patterns of files to skip.
    pub exclude: Vec<String>,
    /// Remove remote files (within the upload scope) that are absent locally.
    pub delete: bool,
    /// Report what would change without uploading anything.
    pub dry_run: bool,
    /// Upload every file, ignoring the remote manifest.
    pub force: bool,
    /// Show a progress spinner.
    pub progress: bool,
}

impl UploadOptions {
    /// Minimal options to upload `local` to `repo_id`, creating it if needed.
    #[must_use]
    pub fn new(repo_id: impl Into<String>, kind: RepoKind, local: Vec<PathBuf>) -> Self {
        Self {
            repo_id: repo_id.into(),
            kind,
            local,
            path_in_repo: None,
            revision: None,
            commit_message: None,
            create: true,
            private: false,
            backends: Vec::new(),
            include: Vec::new(),
            exclude: Vec::new(),
            delete: false,
            dry_run: false,
            force: false,
            progress: false,
        }
    }
}

/// Local file identity used to diff against the remote manifest.
#[derive(Clone, Debug)]
pub(crate) struct LocalHash {
    pub size: u64,
    pub sha256: String,
    pub git_blob_id: String,
}

/// Result of diffing the local tree against a remote revision.
#[derive(Clone, Debug, Default)]
pub(crate) struct Diff {
    pub added: Vec<LocalFile>,
    pub modified: Vec<LocalFile>,
    pub unchanged: usize,
    pub orphans: Vec<String>,
}

impl Diff {
    /// Files that need uploading (added plus modified).
    pub(crate) fn to_upload(&self) -> impl Iterator<Item = &LocalFile> {
        self.added.iter().chain(self.modified.iter())
    }
}

/// Diff `files` against `remote`, considering `allow_delete` for orphans.
pub(crate) fn plan(
    files: &[LocalFile],
    hashes: &BTreeMap<String, LocalHash>,
    remote: &BTreeMap<String, RemoteEntry>,
    scope: Option<&str>,
    allow_delete: bool,
) -> Diff {
    let mut diff = Diff::default();
    let local: BTreeSet<&str> = files.iter().map(|file| file.relative.as_str()).collect();
    for file in files {
        match (hashes.get(&file.relative), remote.get(&file.relative)) {
            (_, None) => diff.added.push(file.clone()),
            (Some(hash), Some(entry)) if same_as(hash, entry) => diff.unchanged += 1,
            _ => diff.modified.push(file.clone()),
        }
    }
    if allow_delete {
        for path in remote.keys() {
            if in_scope(path, scope) && !local.contains(path.as_str()) {
                diff.orphans.push(path.clone());
            }
        }
        diff.orphans.sort();
    }
    diff
}

/// Whether a local file is byte-identical to its remote counterpart.
fn same_as(local: &LocalHash, remote: &RemoteEntry) -> bool {
    if local.size != remote.size {
        return false;
    }
    if let Some(sha) = remote.sha256.as_deref().filter(|value| !value.is_empty()) {
        return sha == local.sha256;
    }
    if let Some(git) = remote
        .git_blob_id
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        return git == local.git_blob_id;
    }
    // No remote hash to compare; equal sizes are treated as unchanged.
    true
}

/// Normalize `path_in_repo` into a delete scope prefix.
pub(crate) fn scope_of(path_in_repo: Option<&str>) -> Option<&str> {
    path_in_repo
        .map(str::trim)
        .map(|value| value.trim_matches('/'))
        .filter(|value| !value.is_empty())
}

/// Whether a repository path falls inside `scope` (all paths when `None`).
pub(crate) fn in_scope(path: &str, scope: Option<&str>) -> bool {
    match scope {
        Some(scope) if !scope.is_empty() => path == scope || path.starts_with(&format!("{scope}/")),
        _ => true,
    }
}

/// Result of an upload, one entry per targeted backend.
#[derive(Clone, Debug)]
pub struct UploadSummary {
    pub results: Vec<BackendUpload>,
}

/// What one backend did.
#[derive(Clone, Debug)]
pub struct BackendUpload {
    pub backend: UploadBackend,
    pub created: bool,
    pub revision: String,
    /// Commit id or URL, when the server returned one.
    pub commit: Option<String>,
    /// Repository-relative paths that were committed.
    pub uploaded: Vec<String>,
    /// Files the server reported as already present.
    pub skipped: usize,
    /// Total bytes committed.
    pub bytes: u64,
    /// Diff summary for an incremental update.
    pub counts: UploadCounts,
}

/// Local-versus-remote change counts for one backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct UploadCounts {
    pub added: usize,
    pub modified: usize,
    pub unchanged: usize,
    pub deleted: usize,
}

/// One file staged for upload.
#[derive(Clone, Debug)]
pub(crate) struct LocalFile {
    pub path: PathBuf,
    /// Forward-slash path inside the repository.
    pub relative: String,
    pub size: u64,
}

impl LocalFile {
    /// File name without any directory component.
    pub(crate) fn name(&self) -> &str {
        self.relative.rsplit('/').next().unwrap_or(&self.relative)
    }
}

/// Stage `local` paths into repository-relative files, applying glob filters.
pub(crate) fn collect_files(
    local: &[PathBuf],
    path_in_repo: Option<&str>,
    include: &[String],
    exclude: &[String],
) -> Result<Vec<LocalFile>> {
    if local.is_empty() {
        bail!("no local path given");
    }
    let include = build_globs(include)?;
    let exclude = build_globs(exclude)?;
    let prefix = path_in_repo
        .map(|value| value.trim_matches('/').to_owned())
        .filter(|value| !value.is_empty());
    let mut files = Vec::new();
    let single_file = local.len() == 1 && fs::metadata(&local[0]).is_ok_and(|meta| meta.is_file());
    for path in local {
        // A single file with `path_in_repo` treats it as the exact destination
        // path rather than a directory prefix.
        if single_file && let Some(destination) = prefix.clone() {
            let size = fs::metadata(path)
                .with_context(|| format!("cannot read local file {}", path.display()))?
                .len();
            files.push(LocalFile {
                path: path.clone(),
                relative: destination,
                size,
            });
            continue;
        }
        stage_path(path, prefix.as_deref(), &mut files)?;
    }
    files.retain(|file| {
        let hit = |set: &GlobSet| set.is_match(&file.relative) || set.is_match(file.name());
        if exclude.as_ref().is_some_and(hit) {
            return false;
        }
        include.as_ref().is_none_or(hit)
    });
    files.sort_by(|a, b| a.relative.cmp(&b.relative));
    if files.is_empty() {
        bail!("no files to upload after filtering");
    }
    Ok(files)
}

fn stage_path(path: &Path, prefix: Option<&str>, files: &mut Vec<LocalFile>) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("cannot read local path {}", path.display()))?;
    if metadata.is_file() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("file name is not valid UTF-8")?;
        let relative = join_relative(prefix, name);
        files.push(LocalFile {
            path: path.to_path_buf(),
            relative,
            size: metadata.len(),
        });
        return Ok(());
    }
    if !metadata.is_dir() {
        bail!("unsupported local path: {}", path.display());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let child = entry.path();
        if child.is_dir() {
            let rel = entry.path();
            let rel = rel.strip_prefix(path).unwrap_or(&rel).to_path_buf();
            let sub = join_relative(prefix, &rel.to_string_lossy());
            stage_path(&child, Some(&sub), files)?;
        } else if child.is_file() {
            let child_metadata = fs::metadata(&child)?;
            let rel = child.strip_prefix(path).unwrap_or(&child).to_path_buf();
            files.push(LocalFile {
                path: child.clone(),
                relative: join_relative(prefix, &rel.to_string_lossy()),
                size: child_metadata.len(),
            });
        }
    }
    Ok(())
}

/// Join a repository sub-path with a relative name using forward slashes.
fn join_relative(prefix: Option<&str>, name: &str) -> String {
    let name = name.replace('\\', "/");
    match prefix {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}/{name}"),
        _ => name,
    }
}

fn build_globs(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern).with_context(|| format!("invalid glob `{pattern}`"))?);
    }
    Ok(Some(builder.build()?))
}

/// Whether any local input is a directory (needed before enabling deletions).
pub(crate) fn has_directory(local: &[PathBuf]) -> bool {
    local
        .iter()
        .any(|path| fs::metadata(path).is_ok_and(|meta| meta.is_dir()))
}

/// Backends that have credentials available.
pub(crate) fn available_backends() -> Vec<UploadBackend> {
    let mut backends = Vec::new();
    if huggingface::load_token().is_some() {
        backends.push(UploadBackend::HuggingFace);
    }
    if modelscope::load_token().is_some() {
        backends.push(UploadBackend::ModelScope);
    }
    backends
}

/// Progress bar for one backend's upload, or a hidden bar when disabled.
pub(crate) fn upload_progress(
    enabled: bool,
    backend: UploadBackend,
    repo_id: &str,
    total: u64,
) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};
    let bar = if enabled {
        ProgressBar::new(total)
    } else {
        ProgressBar::hidden()
    };
    if let Ok(style) = ProgressStyle::default_bar().template(
        "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {decimal_bytes}/{decimal_total_bytes} • {decimal_bytes_per_sec}",
    ) {
        bar.set_style(style.progress_chars("━━─"));
    }
    bar.set_message(format!("{} • {repo_id}", backend.label()));
    bar
}

/// Validate that a repository path never escapes the repository root.
pub(crate) fn validate_relative(path: &str) -> Result<()> {
    if path.is_empty() || path.starts_with('/') || path.contains("..") || path.contains('\\') {
        bail!("invalid repository path: {path}");
    }
    Ok(())
}

/// Split `repo_id` into `(namespace, name)`.
pub(crate) fn split_repo_id(repo_id: &str) -> Result<(&str, &str)> {
    let (namespace, name) = repo_id
        .split_once('/')
        .with_context(|| format!("repository id must be `org/name`: {repo_id}"))?;
    if namespace.is_empty() || name.is_empty() {
        bail!("repository id must be `org/name`: {repo_id}");
    }
    Ok((namespace, name))
}

#[cfg(test)]
mod tests {
    use super::collect_files;
    use std::fs;
    use std::path::PathBuf;

    struct TempTree(PathBuf);

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(name: &str) -> (TempTree, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "modelhub-upload-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        (TempTree(root.clone()), root)
    }

    #[test]
    fn collects_directory_with_prefix_and_filters() {
        let (_cleanup, root) = fixture("collect");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("model.safetensors"), b"x").unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("sub/data.parquet"), b"y").unwrap();

        let all = collect_files(std::slice::from_ref(&root), Some("dest"), &[], &[]).unwrap();
        let names: Vec<_> = all.iter().map(|f| f.relative.clone()).collect();
        assert_eq!(
            names,
            vec![
                "dest/config.json",
                "dest/model.safetensors",
                "dest/sub/data.parquet"
            ]
        );

        let only_safetensors = collect_files(
            std::slice::from_ref(&root),
            None,
            &["*.safetensors".to_owned()],
            &[],
        )
        .unwrap();
        assert_eq!(only_safetensors.len(), 1);
        assert_eq!(only_safetensors[0].relative, "model.safetensors");

        let excluded = collect_files(
            std::slice::from_ref(&root),
            None,
            &[],
            &["*.json".to_owned(), "sub/*".to_owned()],
        )
        .unwrap();
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].relative, "model.safetensors");
    }

    #[test]
    fn single_file_keeps_its_name() {
        let (_cleanup, root) = fixture("single");
        let file = root.join("weights.safetensors");
        fs::write(&file, b"z").unwrap();
        let files = collect_files(&[file], Some("weights.safetensors"), &[], &[]).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative, "weights.safetensors");
    }

    #[test]
    fn plan_classifies_added_modified_and_orphans() {
        use super::{LocalFile, LocalHash, plan};
        use crate::unified::RemoteEntry;
        use std::collections::BTreeMap;

        let files = vec![
            LocalFile {
                path: "a".into(),
                relative: "keep.json".to_owned(),
                size: 2,
            },
            LocalFile {
                path: "b".into(),
                relative: "change.json".to_owned(),
                size: 3,
            },
            LocalFile {
                path: "c".into(),
                relative: "sub/new.json".to_owned(),
                size: 4,
            },
        ];
        let hashes: BTreeMap<String, LocalHash> = files
            .iter()
            .map(|file| {
                (
                    file.relative.clone(),
                    LocalHash {
                        size: file.size,
                        sha256: format!("sha-{}", file.relative),
                        git_blob_id: String::new(),
                    },
                )
            })
            .collect();
        let mut remote = BTreeMap::new();
        remote.insert(
            "keep.json".to_owned(),
            RemoteEntry {
                size: 2,
                sha256: Some("sha-keep.json".to_owned()),
                git_blob_id: None,
            },
        );
        remote.insert(
            "change.json".to_owned(),
            RemoteEntry {
                size: 3,
                sha256: Some("different".to_owned()),
                git_blob_id: None,
            },
        );
        remote.insert(
            "old.json".to_owned(),
            RemoteEntry {
                size: 1,
                sha256: Some("x".to_owned()),
                git_blob_id: None,
            },
        );

        let diff = plan(&files, &hashes, &remote, None, true);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].relative, "sub/new.json");
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.modified[0].relative, "change.json");
        assert_eq!(diff.unchanged, 1);
        assert_eq!(diff.orphans, vec!["old.json"]);

        // Scoping deletions to `sub` keeps `old.json`.
        let scoped = plan(&files, &hashes, &remote, Some("sub"), true);
        assert!(scoped.orphans.is_empty());
        // Without `allow_delete` nothing is pruned.
        let no_delete = plan(&files, &hashes, &remote, None, false);
        assert!(no_delete.orphans.is_empty());
    }
}
