//! Verified, content-addressed downloads shared across model backends.

use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub use crate::repos::RepoKind;

const HF_MIRROR: &str = "https://hf-mirror.com";
const HF_OFFICIAL: &str = "https://huggingface.co";
const DATASET_PAGE_SIZE: usize = 200;
const MS_OFFICIAL: &str = "https://modelscope.cn";

/// Records the files a completed download expects, so completeness can be
/// verified offline.
pub const MANIFEST_FILE: &str = ".modelhub-manifest.json";

/// Hub that hosts a repository.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Hugging Face hub.
    HuggingFace,
    /// `ModelScope` hub.
    ModelScope,
}

impl Backend {
    /// URL segment and cache directory name for this backend.
    #[must_use]
    pub const fn segment(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }

    /// Revision requested when the caller does not name one.
    #[must_use]
    pub const fn default_revision(self) -> &'static str {
        match self {
            Self::HuggingFace => "main",
            Self::ModelScope => "master",
        }
    }
}

/// On-disk record of a completed modelhub download.
///
/// `list --check` compares the cached snapshots against this record without any
/// network access.
#[derive(Debug, Deserialize, Serialize)]
pub struct RepoManifest {
    pub version: u32,
    pub kind: RepoKind,
    pub backends: BTreeMap<String, BackendManifest>,
}

/// Expected files for one backend snapshot.
#[derive(Debug, Deserialize, Serialize)]
pub struct BackendManifest {
    pub revision: String,
    /// Relative repository path to expected file size in bytes.
    pub files: BTreeMap<String, u64>,
}

impl BackendManifest {
    fn from_manifest(manifest: &Manifest) -> Self {
        Self {
            revision: manifest.revision.clone(),
            files: manifest
                .files
                .iter()
                .map(|(path, file)| (path.clone(), file.size))
                .collect(),
        }
    }
}

/// Read the recorded manifest for a modelhub-owned repository, if present.
#[must_use]
pub fn read_repo_manifest(repo_root: &Path) -> Option<RepoManifest> {
    let data = fs::read(repo_root.join(MANIFEST_FILE)).ok()?;
    serde_json::from_slice(&data).ok()
}

/// One remote file's identity, used to diff a local tree against the Hub.
#[derive(Clone, Debug)]
pub struct RemoteEntry {
    pub size: u64,
    pub sha256: Option<String>,
    pub git_blob_id: Option<String>,
}

/// Remote file list for a repository revision.
#[derive(Clone, Debug)]
pub struct RemoteManifest {
    pub files: BTreeMap<String, RemoteEntry>,
}

impl RemoteManifest {
    fn from_manifest(manifest: &Manifest) -> Self {
        Self {
            files: manifest
                .files
                .iter()
                .map(|(path, file)| {
                    (
                        path.clone(),
                        RemoteEntry {
                            size: file.size,
                            sha256: file.sha256.clone(),
                            git_blob_id: file.git_blob_id.clone(),
                        },
                    )
                })
                .collect(),
        }
    }
}

/// Fetch the current remote file list for one backend and revision.
///
/// Used by `upload` to upload only changed files.
pub async fn fetch_remote_manifest(
    backend: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<RemoteManifest> {
    let manifest = match backend {
        "huggingface" => {
            let client = huggingface_client()?;
            huggingface_manifest(&client, kind, repo_id, revision).await?
        }
        "modelscope" => {
            let client = crate::modelscope::client::http_client()?;
            modelscope_manifest(&client, kind, repo_id, revision).await?
        }
        other => bail!("unsupported backend: {other}"),
    };
    Ok(RemoteManifest::from_manifest(&manifest))
}

/// Persist a manifest record beside a repository's snapshots.
pub fn write_repo_manifest_record(repo_root: &Path, record: &RepoManifest) -> anyhow::Result<()> {
    let data = serde_json::to_vec_pretty(record)?;
    fs::write(repo_root.join(MANIFEST_FILE), data)?;
    Ok(())
}

/// Fetch one backend's current file list for a repository revision.
///
/// `backend` is `"huggingface"` or `"modelscope"`. Used by `check` to build a
/// manifest for repositories modelhub did not download itself.
pub async fn fetch_backend_manifest(
    backend: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<BackendManifest> {
    match backend {
        "huggingface" => {
            let client = huggingface_client()?;
            let manifest = huggingface_manifest(&client, kind, repo_id, revision).await?;
            Ok(BackendManifest::from_manifest(&manifest))
        }
        "modelscope" => {
            let client = crate::modelscope::client::http_client()?;
            let manifest = modelscope_manifest(&client, kind, repo_id, revision).await?;
            Ok(BackendManifest::from_manifest(&manifest))
        }
        other => bail!("unsupported backend: {other}"),
    }
}

fn write_repo_manifest(
    repo_root: &Path,
    kind: RepoKind,
    hf: Option<&Manifest>,
    ms: Option<&Manifest>,
) -> anyhow::Result<()> {
    let mut backends = BTreeMap::new();
    if let Some(manifest) = hf {
        backends.insert(
            Backend::HuggingFace.segment().to_owned(),
            BackendManifest::from_manifest(manifest),
        );
    }
    if let Some(manifest) = ms {
        backends.insert(
            Backend::ModelScope.segment().to_owned(),
            BackendManifest::from_manifest(manifest),
        );
    }
    let record = RepoManifest {
        version: 1,
        kind,
        backends,
    };
    let data = serde_json::to_vec_pretty(&record)?;
    fs::write(repo_root.join(MANIFEST_FILE), data)?;
    Ok(())
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug)]
struct RemoteFile {
    backend: Backend,
    path: String,
    size: u64,
    sha256: Option<String>,
    git_blob_id: Option<String>,
    url: String,
}

#[derive(Debug)]
struct Manifest {
    revision: String,
    files: BTreeMap<String, RemoteFile>,
}

/// Backend-specific snapshot roots created from one content-addressed blob store.
#[derive(Debug)]
pub struct DownloadedRepo {
    /// Whether a model or a dataset was downloaded.
    pub kind: RepoKind,
    /// modelhub-owned root, e.g. `<cache>/datasets/<ns--name>`.
    pub repo_root: PathBuf,
    /// Hugging Face snapshot root, when that backend had the repository.
    pub huggingface_root: Option<PathBuf>,
    /// `ModelScope` snapshot root, when that backend had the repository.
    pub modelscope_root: Option<PathBuf>,
    /// Exact file materialized for a single-file download.
    pub file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct HfInfo {
    sha: String,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(rename = "blobId")]
    #[serde(default)]
    blob_id: Option<String>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    sha256: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct MsResponse {
    // Model file listings send `Success`; dataset tree listings omit it and
    // signal failure through `Data: null` or a non-2xx status instead.
    #[serde(rename = "Success", default = "default_true")]
    success: bool,
    #[serde(rename = "Message", default)]
    message: String,
    #[serde(rename = "Data")]
    data: Option<MsData>,
}

#[derive(Debug, Deserialize)]
struct MsData {
    #[serde(rename = "Files")]
    files: Vec<MsFile>,
}

#[derive(Debug, Deserialize)]
struct MsFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size", default)]
    size: u64,
    #[serde(rename = "Sha256", default)]
    sha256: Option<String>,
    #[serde(rename = "Type", default)]
    file_type: String,
}

#[derive(Debug)]
struct Artifact {
    remote: RemoteFile,
    targets: Vec<(Backend, String)>,
}

fn encode_path(value: &str) -> String {
    value
        .split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn hf_endpoints() -> Vec<String> {
    std::env::var("HF_ENDPOINT").map_or_else(
        |_| vec![HF_MIRROR.to_owned(), HF_OFFICIAL.to_owned()],
        |value| vec![value.trim_end_matches('/').to_owned()],
    )
}

/// Base URL for `ModelScope` requests; `MODELSCOPE_ENDPOINT` overrides it.
fn ms_base_url() -> String {
    std::env::var("MODELSCOPE_ENDPOINT")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| MS_OFFICIAL.to_owned())
}

/// URL path prefix for a repository kind on Hugging Face.
///
/// Models are served from the site root; datasets live under `datasets/`.
fn hf_repo_prefix(kind: RepoKind) -> &'static str {
    match kind {
        RepoKind::Model => "",
        RepoKind::Dataset => "datasets/",
    }
}

/// Resolve URL for one file on one Hugging Face endpoint.
fn huggingface_file_url(
    endpoint: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
    file: &str,
) -> String {
    format!(
        "{endpoint}/{}{}/resolve/{}/{}",
        hf_repo_prefix(kind),
        encode_path(repo_id),
        encode_path(revision),
        encode_path(file)
    )
}

/// Single-file download URL on `ModelScope`.
fn modelscope_file_url(
    base: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
    file: &str,
) -> String {
    format!(
        "{base}/api/v1/{}/{}/repo?Revision={}&FilePath={}",
        kind.segment(),
        repo_id,
        urlencoding::encode(revision),
        urlencoding::encode(file)
    )
}

fn hf_auth(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN")) {
        Ok(token) => request.bearer_auth(token),
        Err(_) => request,
    }
}

async fn huggingface_manifest(
    client: &reqwest::Client,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Manifest> {
    let mut errors = Vec::new();
    for endpoint in hf_endpoints() {
        let url = format!(
            "{endpoint}/api/{}/{}/revision/{}?blobs=true",
            kind.segment(),
            encode_path(repo_id),
            encode_path(revision)
        );
        match hf_auth(client.get(url)).send().await {
            Ok(response) if response.status().is_success() => {
                let info = response.json::<HfInfo>().await?;
                let files = info
                    .siblings
                    .into_iter()
                    .map(|file| {
                        let is_lfs = file.lfs.is_some();
                        let size = file.size.or_else(|| file.lfs.as_ref().map(|lfs| lfs.size));
                        let sha256 = file.lfs.map(|lfs| lfs.sha256);
                        let path = file.rfilename;
                        let remote = RemoteFile {
                            backend: Backend::HuggingFace,
                            size: size.unwrap_or(0),
                            sha256,
                            git_blob_id: if is_lfs { None } else { file.blob_id },
                            url: huggingface_file_url(&endpoint, kind, repo_id, &info.sha, &path),
                            path: path.clone(),
                        };
                        (path, remote)
                    })
                    .collect();
                return Ok(Manifest {
                    revision: info.sha,
                    files,
                });
            }
            Ok(response) => errors.push(format!("{endpoint}: HTTP {}", response.status())),
            Err(error) => errors.push(format!("{endpoint}: {error}")),
        }
    }
    bail!("Hugging Face manifest failed: {}", errors.join("; "))
}

/// List every file in a repository on one supported backend.
///
/// Models are listed with a single recursive request; datasets use the
/// paginated tree endpoint and are accumulated page by page.
async fn modelscope_manifest(
    client: &reqwest::Client,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Manifest> {
    let files = modelscope_list_files(client, kind, repo_id, revision).await?;
    let files = files
        .into_iter()
        .filter(|file| file.file_type != "tree")
        .map(|file| {
            let path = file.path;
            let remote = RemoteFile {
                backend: Backend::ModelScope,
                size: file.size,
                sha256: file.sha256.filter(|hash| !hash.is_empty()),
                git_blob_id: None,
                url: modelscope_file_url(&ms_base_url(), kind, repo_id, revision, &path),
                path: path.clone(),
            };
            (path, remote)
        })
        .collect();
    Ok(Manifest {
        revision: revision.to_owned(),
        files,
    })
}

async fn modelscope_list_files(
    client: &reqwest::Client,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<MsFile>> {
    match kind {
        RepoKind::Model => {
            let url = format!(
                "{}/api/v1/models/{repo_id}/repo/files?Recursive=true&Revision={}",
                ms_base_url(),
                urlencoding::encode(revision)
            );
            let response = client.get(url).send().await?;
            modelscope_files_from_response(response).await
        }
        RepoKind::Dataset => {
            let mut all = Vec::new();
            for page in 1usize.. {
                let url = format!(
                    "{}/api/v1/datasets/{repo_id}/repo/tree?Recursive=True&Revision={}&PageNumber={page}&PageSize={DATASET_PAGE_SIZE}",
                    ms_base_url(),
                    urlencoding::encode(revision)
                );
                let response = client.get(url).send().await?;
                let mut files = modelscope_files_from_response(response).await?;
                let count = files.len();
                all.append(&mut files);
                if count < DATASET_PAGE_SIZE {
                    break;
                }
            }
            Ok(all)
        }
    }
}

async fn modelscope_files_from_response(
    response: reqwest::Response,
) -> anyhow::Result<Vec<MsFile>> {
    if !response.status().is_success() {
        bail!("ModelScope manifest failed: HTTP {}", response.status());
    }
    let parsed = response.json::<MsResponse>().await?;
    if !parsed.success {
        bail!("ModelScope manifest failed: {}", parsed.message);
    }
    Ok(parsed
        .data
        .context("ModelScope manifest did not include data")?
        .files)
}

/// Reject empty, absolute, or `..`-containing repository paths and revisions.
///
/// Callers run this before any network request so a bad `file` or `revision`
/// never reaches a hub. Unlike `upload::validate_relative`, this is
/// component-based: a name like `file..txt` is allowed, and a backslash is an
/// ordinary character on Unix (Windows path prefixes are still rejected).
fn validate_repo_value(label: &str, value: &str) -> anyhow::Result<()> {
    let path = Path::new(value);
    let invalid = value.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
    if invalid {
        bail!("{label} must be a non-empty relative path without `..`: {value}");
    }
    Ok(())
}

fn safe_path(root: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!("repository path must be relative: {}", path.display());
    }
    let mut output = root.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(value) => output.push(value),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("repository path escapes cache: {}", path.display());
            }
        }
    }
    Ok(output)
}

fn is_critical(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let critical_extension = Path::new(&lower).extension().is_some_and(|extension| {
        ["safetensors", "bin", "gguf", "onnx"]
            .iter()
            .any(|value| extension.eq_ignore_ascii_case(value))
    });
    critical_extension || lower.ends_with("tokenizer.json") || lower.ends_with("tokenizer.model")
}

fn digest_file(path: &Path, size: u64) -> anyhow::Result<(String, String)> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut sha256 = Sha256::new();
    let mut git = Sha1::new();
    git.update(format!("blob {size}\0").as_bytes());
    let mut buffer = vec![0u8; 64 * 1024].into_boxed_slice();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        sha256.update(&buffer[..count]);
        git.update(&buffer[..count]);
    }
    Ok((
        format!("{:x}", sha256.finalize()),
        format!("{:x}", git.finalize()),
    ))
}

async fn materialize(
    remote: &RemoteFile,
    cache_root: &Path,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    progress: &ProgressBar,
) -> anyhow::Result<(PathBuf, String, String)> {
    if let Some(hash) = remote.sha256.as_ref() {
        let cached = cache_root.join("blobs").join("sha256").join(hash);
        if cached.is_file() {
            let size = fs::metadata(&cached)?.len();
            progress.inc(size);
            let (_, git) = digest_file(&cached, size)?;
            return Ok((cached, hash.clone(), git));
        }
    }
    validate_repo_value("repository path", &remote.path)?;
    let request = match remote.backend {
        Backend::HuggingFace => hf_auth(hf_client.get(&remote.url)),
        Backend::ModelScope => ms_client.get(&remote.url).header(
            crate::modelscope::client::USER_AGENT.0,
            crate::modelscope::client::USER_AGENT.1,
        ),
    };
    let response = request.send().await?;
    if !response.status().is_success() {
        bail!(
            "failed to download {}: HTTP {}",
            remote.path,
            response.status()
        );
    }
    stream_to_blob(
        remote.backend,
        &remote.path,
        response,
        Some(remote),
        cache_root,
        progress,
    )
    .await
}

/// Removes a staging file when dropped unless the download completed.
struct StagingGuard {
    path: PathBuf,
    keep: bool,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Stream a response into staging, verify it, and move it into the
/// content-addressed blob store.
///
/// `expected` carries the manifest checks for a repository download. A
/// single-file download passes `None` and relies on the response's
/// `Content-Length` instead. When neither a manifest size nor `Content-Length`
/// is available (for example a chunked response), the Git prelude falls back to
/// zero and a Git blob comparison will not match.
async fn stream_to_blob(
    backend: Backend,
    path: &str,
    response: reqwest::Response,
    expected: Option<&RemoteFile>,
    cache_root: &Path,
    progress: &ProgressBar,
) -> anyhow::Result<(PathBuf, String, String)> {
    let declared = response.content_length().filter(|length| *length > 0);
    let staging_target = safe_path(&cache_root.join("staging").join(backend.segment()), path)?;
    let staging_name = staging_target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    let staging = staging_target.with_file_name(format!("{staging_name}.part"));
    if let Some(parent) = staging.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut guard = StagingGuard {
        path: staging.clone(),
        keep: false,
    };
    let expected_size = expected
        .map(|remote| remote.size)
        .filter(|size| *size > 0)
        .or(declared);
    let mut sha256 = Sha256::new();
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", expected_size.unwrap_or(0)).as_bytes());
    let mut written = 0u64;
    {
        let mut writer = BufWriter::new(fs::File::create(&staging)?);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            written += chunk.len() as u64;
            progress.inc(chunk.len() as u64);
            sha256.update(&chunk);
            git.update(&chunk);
            writer.write_all(&chunk)?;
        }
        writer.flush()?;
    }
    if let Some(size) = expected_size
        && written != size
    {
        bail!("incomplete download for {path}");
    }
    let sha256 = format!("{:x}", sha256.finalize());
    let git = format!("{:x}", git.finalize());
    if let Some(expected) = expected.and_then(|remote| remote.sha256.as_deref())
        && expected != sha256
    {
        bail!("SHA-256 mismatch for {path}");
    }
    if let Some(expected) = expected.and_then(|remote| remote.git_blob_id.as_deref())
        && expected != git
    {
        bail!("Git blob hash mismatch for {path}");
    }
    let blob = cache_root.join("blobs").join("sha256").join(&sha256);
    if let Some(parent) = blob.parent() {
        fs::create_dir_all(parent)?;
    }
    if blob.exists() {
        fs::remove_file(&staging)?;
    } else if let Err(error) = fs::rename(&staging, &blob) {
        // On Windows a concurrent download may have produced the same
        // content-addressed blob between the existence check and the rename; in
        // that case the already-complete blob wins and this staging file is
        // discarded. Otherwise the rename error is real.
        if blob.exists() {
            fs::remove_file(&staging)?;
        } else {
            return Err(error.into());
        }
    }
    guard.keep = true;
    Ok((blob, sha256, git))
}

fn link_artifact(blob: &Path, target: &Path) -> anyhow::Result<()> {
    if target.exists() {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::hard_link(blob, target)
        .or_else(|_| fs::copy(blob, target).map(|_| ()))
        .with_context(|| format!("failed to materialize {}", target.display()))
}

/// Spinner shown while downloading, hidden when the caller disabled progress.
fn progress_bar(enabled: bool, message: String) -> anyhow::Result<ProgressBar> {
    let progress = if enabled {
        ProgressBar::new_spinner()
    } else {
        ProgressBar::hidden()
    };
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}")?,
    );
    progress.set_message(message);
    progress.enable_steady_tick(std::time::Duration::from_millis(100));
    Ok(progress)
}

fn snapshot_root(model_root: &Path, backend: Backend, revision: &str) -> PathBuf {
    model_root
        .join(match backend {
            Backend::HuggingFace => "huggingface",
            Backend::ModelScope => "modelscope",
        })
        .join("snapshots")
        .join(revision)
}

fn add_separate(plans: &mut Vec<Artifact>, file: RemoteFile) {
    plans.push(Artifact {
        targets: vec![(file.backend, file.path.clone())],
        remote: file,
    });
}

/// URLs to try for one file, one per endpoint of `backend`.
fn single_file_urls(
    backend: Backend,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
    file: &str,
) -> Vec<String> {
    match backend {
        Backend::HuggingFace => hf_endpoints()
            .into_iter()
            .map(|endpoint| huggingface_file_url(&endpoint, kind, repo_id, revision, file))
            .collect(),
        Backend::ModelScope => {
            vec![modelscope_file_url(
                &ms_base_url(),
                kind,
                repo_id,
                revision,
                file,
            )]
        }
    }
}

/// Assemble the result for a single-file download from the winning backend.
fn single_file_result(
    kind: RepoKind,
    backend: Backend,
    repo_root: PathBuf,
    file: PathBuf,
) -> DownloadedRepo {
    let root = repo_root.join(backend.segment());
    DownloadedRepo {
        kind,
        huggingface_root: (backend == Backend::HuggingFace).then_some(root.clone()),
        modelscope_root: (backend == Backend::ModelScope).then_some(root),
        repo_root,
        file: Some(file),
    }
}

/// Fire every candidate request and return the first successful response.
///
/// Dropping the remaining futures cancels their requests: a miss costs one
/// failed request per candidate and never a repository listing.
async fn send_first_success(
    candidates: Vec<(RepoKind, Backend, String)>,
    file: &str,
    repo_id: &str,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
) -> anyhow::Result<(RepoKind, Backend, String, reqwest::Response)> {
    let attempts = candidates.len().max(1);
    let requests =
        futures_util::stream::iter(candidates.into_iter().map(|(kind, backend, url)| {
            let hf_client = hf_client.clone();
            let ms_client = ms_client.clone();
            async move {
                let request = match backend {
                    Backend::HuggingFace => hf_auth(hf_client.get(&url)),
                    Backend::ModelScope => ms_client.get(&url).header(
                        crate::modelscope::client::USER_AGENT.0,
                        crate::modelscope::client::USER_AGENT.1,
                    ),
                };
                let response = request
                    .send()
                    .await
                    .map_err(|error| format!("{url}: {error}"))?;
                if !response.status().is_success() {
                    return Err(format!("{url}: HTTP {}", response.status()));
                }
                Ok((kind, backend, url, response))
            }
        }))
        .buffer_unordered(attempts);
    futures_util::pin_mut!(requests);
    let mut errors = Vec::new();
    while let Some(result) = requests.next().await {
        match result {
            Ok(winner) => return Ok(winner),
            Err(error) => errors.push(error),
        }
    }
    bail!(
        "`{file}` was not found in `{repo_id}`; tried: {}",
        errors.join("; ")
    )
}

/// Write the marker files that let `list` and `clear` identify a repository.
fn ensure_repo_markers(repo_root: &Path, repo_id: &str) -> anyhow::Result<()> {
    fs::create_dir_all(repo_root)?;
    fs::write(repo_root.join(".modelhub-model-id"), repo_id)?;
    fs::write(repo_root.join(".modelhub-layout"), "cas-v1")?;
    Ok(())
}

/// Snapshot revisions that may already hold `file` for one candidate.
///
/// A whole-repository Hugging Face download stores its snapshots under the
/// resolved commit sha and records that sha in `refs/<revision>`. Consulting the
/// local ref lets an existing repository download satisfy a single-file request
/// without touching the network.
fn snapshot_revisions(repo_root: &Path, backend: Backend, revision: &str) -> Vec<String> {
    let mut revisions = vec![revision.to_owned()];
    if backend == Backend::HuggingFace
        && let Ok(sha) =
            fs::read_to_string(repo_root.join("huggingface").join("refs").join(revision))
    {
        let sha = sha.trim();
        if !sha.is_empty() && sha != revision {
            revisions.push(sha.to_owned());
        }
    }
    revisions
}

/// Download one repository file into the modelhub cache.
///
/// Never lists the repository: the cache is checked first, and a miss issues one
/// request per candidate (`kind` × `backend`, narrowed by the hints). The first
/// successful response wins and the remaining requests are cancelled. The file
/// stays in the modelhub cache and is not linked into native backend caches.
pub async fn download_single_file(
    kind: Option<RepoKind>,
    backend: Option<Backend>,
    repo_id: &str,
    revision: Option<&str>,
    file: &str,
    cache_root: &Path,
    progress: bool,
) -> anyhow::Result<DownloadedRepo> {
    validate_repo_value("repo id", repo_id)?;
    validate_repo_value("file path", file)?;
    if let Some(revision) = revision {
        validate_repo_value("revision", revision)?;
    }
    let kinds = kind.map_or_else(
        || vec![RepoKind::Model, RepoKind::Dataset],
        |kind| vec![kind],
    );
    let backends = backend.map_or_else(
        || vec![Backend::HuggingFace, Backend::ModelScope],
        |backend| vec![backend],
    );
    let repo_root = |kind: RepoKind| {
        cache_root
            .join(kind.segment())
            .join(repo_id.replace('/', "--"))
    };
    for kind in &kinds {
        for backend in &backends {
            let revision =
                revision.map_or_else(|| backend.default_revision().to_owned(), str::to_owned);
            for snapshot_revision in snapshot_revisions(&repo_root(*kind), *backend, &revision) {
                let target = safe_path(
                    &snapshot_root(&repo_root(*kind), *backend, &snapshot_revision),
                    file,
                )?;
                if target.is_file() {
                    ensure_repo_markers(&repo_root(*kind), repo_id)?;
                    return Ok(single_file_result(
                        *kind,
                        *backend,
                        repo_root(*kind),
                        target,
                    ));
                }
            }
        }
    }
    let mut candidates = Vec::new();
    for kind in &kinds {
        for backend in &backends {
            let revision =
                revision.map_or_else(|| backend.default_revision().to_owned(), str::to_owned);
            for url in single_file_urls(*backend, *kind, repo_id, &revision, file) {
                candidates.push((*kind, *backend, url));
            }
        }
    }
    let hf_client = huggingface_client()?;
    let ms_client = crate::modelscope::client::http_client()?;
    let bar = progress_bar(progress, format!("{repo_id} • downloading {file}"))?;
    let outcome = async {
        let (kind, backend, _, response) =
            send_first_success(candidates, file, repo_id, &hf_client, &ms_client).await?;
        let (blob, _, _) = stream_to_blob(backend, file, response, None, cache_root, &bar).await?;
        ensure_repo_markers(&repo_root(kind), repo_id)?;
        let revision =
            revision.map_or_else(|| backend.default_revision().to_owned(), str::to_owned);
        let target = safe_path(&snapshot_root(&repo_root(kind), backend, &revision), file)?;
        link_artifact(&blob, &target)?;
        Ok::<_, anyhow::Error>(single_file_result(kind, backend, repo_root(kind), target))
    }
    .await;
    match outcome {
        Ok(downloaded) => {
            bar.finish_with_message(format!("✓ {repo_id} • downloaded {file}"));
            Ok(downloaded)
        }
        Err(error) => {
            bar.abandon_with_message(format!("{repo_id} • download failed"));
            Err(error)
        }
    }
}

/// Download a repository, auto-detecting whether it is a model or a dataset.
///
/// `file` selects a single repository file; `None` downloads the whole repo.
/// `progress` toggles the progress bar.
#[allow(clippy::too_many_arguments)]
pub async fn download_repo(
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
    cache_root: &Path,
    concurrency: usize,
    allow_weight_mismatch: bool,
    file: Option<&str>,
    progress: bool,
) -> anyhow::Result<DownloadedRepo> {
    let hf_client = huggingface_client()?;
    let ms_client = crate::modelscope::client::http_client()?;
    let (kind, hf, ms) = detect_manifests(
        &hf_client,
        &ms_client,
        repo_id,
        huggingface_revision,
        modelscope_revision,
    )
    .await?;
    download_loaded(
        kind,
        repo_id,
        hf,
        ms,
        huggingface_revision,
        cache_root,
        &hf_client,
        &ms_client,
        concurrency,
        allow_weight_mismatch,
        file,
        progress,
    )
    .await
}

fn huggingface_client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?)
}

/// Quietly probe model and dataset manifests to decide what `repo_id` is.
///
/// A repository that exists as both a model and a dataset is rejected so the
/// caller does not have to guess which one was meant.
async fn detect_manifests(
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
) -> anyhow::Result<(RepoKind, Option<Manifest>, Option<Manifest>)> {
    let (model, dataset) = tokio::join!(
        probe_manifests(
            RepoKind::Model,
            hf_client,
            ms_client,
            repo_id,
            huggingface_revision,
            modelscope_revision
        ),
        probe_manifests(
            RepoKind::Dataset,
            hf_client,
            ms_client,
            repo_id,
            huggingface_revision,
            modelscope_revision
        )
    );
    let (model_hf, model_ms) = model;
    let (dataset_hf, dataset_ms) = dataset;
    let model_found = model_hf.is_some() || model_ms.is_some();
    let dataset_found = dataset_hf.is_some() || dataset_ms.is_some();
    if model_found && dataset_found {
        bail!(
            "`{repo_id}` exists as both a model and a dataset on supported backends; \
             use a distinct repository ID"
        );
    }
    if model_found {
        return Ok((RepoKind::Model, model_hf, model_ms));
    }
    if dataset_found {
        return Ok((RepoKind::Dataset, dataset_hf, dataset_ms));
    }
    bail!(
        "`{repo_id}` was not found as a model or a dataset on any supported backend; \
         check the ID, revision, and credentials (HF_TOKEN for Hugging Face)"
    );
}

/// Probe one kind on both backends, discarding error details.
async fn probe_manifests(
    kind: RepoKind,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
) -> (Option<Manifest>, Option<Manifest>) {
    let (hf, ms) = tokio::join!(
        huggingface_manifest(hf_client, kind, repo_id, huggingface_revision),
        modelscope_manifest(ms_client, kind, repo_id, modelscope_revision)
    );
    (hf.ok(), ms.ok())
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn download_loaded(
    kind: RepoKind,
    repo_id: &str,
    hf: Option<Manifest>,
    ms: Option<Manifest>,
    huggingface_revision: &str,
    cache_root: &Path,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    concurrency: usize,
    allow_weight_mismatch: bool,
    file: Option<&str>,
    progress: bool,
) -> anyhow::Result<DownloadedRepo> {
    if concurrency == 0 {
        bail!("download concurrency must be at least 1");
    }
    let repo_root = cache_root
        .join(kind.segment())
        .join(repo_id.replace('/', "--"));
    fs::create_dir_all(&repo_root)?;
    fs::write(repo_root.join(".modelhub-model-id"), repo_id)?;
    fs::write(repo_root.join(".modelhub-layout"), "cas-v1")?;
    let mut paths = BTreeSet::new();
    if let Some(manifest) = hf.as_ref() {
        paths.extend(manifest.files.keys().cloned());
    }
    if let Some(manifest) = ms.as_ref() {
        paths.extend(manifest.files.keys().cloned());
    }
    let file = file.map(|file| file.trim_start_matches('/').to_owned());
    if let Some(file) = file.as_deref() {
        if !paths.contains(file) {
            bail!("`{file}` is not present in {} `{repo_id}`", kind.label());
        }
        paths.retain(|path| path == file);
    }
    let mut plans = Vec::new();
    let mut git_comparisons = Vec::new();
    let mut mismatches = Vec::new();
    for path in paths {
        let hf_file = hf
            .as_ref()
            .and_then(|manifest| manifest.files.get(&path))
            .cloned();
        let ms_file = ms
            .as_ref()
            .and_then(|manifest| manifest.files.get(&path))
            .cloned();
        match (hf_file, ms_file) {
            (Some(hf_file), Some(ms_file)) => {
                if hf_file.size != ms_file.size
                    || matches!((&hf_file.sha256, &ms_file.sha256), (Some(a), Some(b)) if a != b)
                {
                    if kind == RepoKind::Model && is_critical(&path) && !allow_weight_mismatch {
                        mismatches.push(path);
                    } else {
                        add_separate(&mut plans, hf_file);
                        add_separate(&mut plans, ms_file);
                    }
                } else if hf_file.sha256.is_some() && hf_file.sha256 == ms_file.sha256 {
                    plans.push(Artifact {
                        remote: hf_file,
                        targets: vec![
                            (Backend::HuggingFace, path.clone()),
                            (Backend::ModelScope, path),
                        ],
                    });
                } else {
                    git_comparisons.push((hf_file, ms_file));
                }
            }
            (Some(file), None) | (None, Some(file)) => add_separate(&mut plans, file),
            (None, None) => {}
        }
    }
    if !mismatches.is_empty() {
        bail!(
            "critical files differ across backends: {}; rerun with --all-backends to keep both versions",
            mismatches.join(", ")
        );
    }
    let hf_root = hf.as_ref().map(|_| repo_root.join("huggingface"));
    let ms_root = ms.as_ref().map(|_| repo_root.join("modelscope"));
    let hf_snapshot_revision = hf.as_ref().map(|manifest| manifest.revision.clone());
    let ms_snapshot_revision = ms.as_ref().map(|manifest| manifest.revision.clone());
    let progress = progress_bar(progress, format!("{repo_id} • verifying and downloading"))?;
    for (hf_file, ms_file) in git_comparisons {
        let (blob, sha256, git) =
            materialize(&ms_file, cache_root, hf_client, ms_client, &progress).await?;
        let ms_target = safe_path(
            &snapshot_root(
                &repo_root,
                Backend::ModelScope,
                ms_snapshot_revision
                    .as_deref()
                    .context("missing ModelScope revision")?,
            ),
            &ms_file.path,
        )?;
        link_artifact(&blob, &ms_target)?;
        let hashes_match = hf_file
            .sha256
            .as_ref()
            .is_some_and(|expected| expected == &sha256)
            || hf_file.git_blob_id.as_ref() == Some(&git);
        if hashes_match {
            let hf_target = safe_path(
                &snapshot_root(
                    &repo_root,
                    Backend::HuggingFace,
                    hf_snapshot_revision
                        .as_deref()
                        .context("missing Hugging Face revision")?,
                ),
                &hf_file.path,
            )?;
            link_artifact(&blob, &hf_target)?;
        } else if kind == RepoKind::Model && is_critical(&hf_file.path) && !allow_weight_mismatch {
            bail!("critical file differs across backends: {}", hf_file.path);
        } else {
            add_separate(&mut plans, hf_file);
        }
    }
    let file_path = match file.as_deref() {
        Some(file)
            if hf
                .as_ref()
                .is_some_and(|manifest| manifest.files.contains_key(file)) =>
        {
            let revision = hf_snapshot_revision
                .as_deref()
                .context("missing Hugging Face revision")?;
            Some(safe_path(
                &snapshot_root(&repo_root, Backend::HuggingFace, revision),
                file,
            )?)
        }
        Some(file) => {
            let revision = ms_snapshot_revision
                .as_deref()
                .context("missing ModelScope revision")?;
            Some(safe_path(
                &snapshot_root(&repo_root, Backend::ModelScope, revision),
                file,
            )?)
        }
        None => None,
    };
    let hf_client = Arc::new(hf_client.clone());
    let ms_client = Arc::new(ms_client.clone());
    let downloads = futures_util::stream::iter(plans.into_iter().map(|artifact| {
        let cache_root = cache_root.to_path_buf();
        let repo_root = repo_root.clone();
        let hf_client = hf_client.clone();
        let ms_client = ms_client.clone();
        let progress = progress.clone();
        let hf_revision = hf_snapshot_revision.clone();
        let ms_revision = ms_snapshot_revision.clone();
        async move {
            let (blob, _, _) = materialize(
                &artifact.remote,
                &cache_root,
                &hf_client,
                &ms_client,
                &progress,
            )
            .await?;
            for (backend, path) in artifact.targets {
                let revision = match backend {
                    Backend::HuggingFace => hf_revision
                        .as_deref()
                        .context("missing Hugging Face revision")?,
                    Backend::ModelScope => ms_revision
                        .as_deref()
                        .context("missing ModelScope revision")?,
                };
                let target = safe_path(&snapshot_root(&repo_root, backend, revision), &path)?;
                link_artifact(&blob, &target)?;
            }
            anyhow::Ok(())
        }
    }))
    .buffer_unordered(concurrency);
    futures_util::pin_mut!(downloads);
    while let Some(result) = downloads.next().await {
        if let Err(error) = result {
            progress.abandon_with_message(format!("{repo_id} • download failed"));
            return Err(error);
        }
    }
    progress.finish_with_message(format!("✓ {repo_id} • verified backend snapshots"));
    if let Some(manifest) = hf.as_ref() {
        let refs = repo_root.join("huggingface").join("refs");
        fs::create_dir_all(&refs)?;
        fs::write(refs.join(huggingface_revision), &manifest.revision)?;
    }
    // Only a full download yields a complete manifest; a single-file download
    // stays unverifiable and reports `unknown` in `list --check`.
    if file.is_none() {
        write_repo_manifest(&repo_root, kind, hf.as_ref(), ms.as_ref())?;
    }
    Ok(DownloadedRepo {
        kind,
        repo_root: repo_root.clone(),
        huggingface_root: hf_root,
        modelscope_root: ms_root,
        file: file_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huggingface_file_url_uses_kind_specific_prefix() {
        assert_eq!(
            huggingface_file_url(
                "https://hf.example",
                RepoKind::Model,
                "org/name",
                "main",
                "data/a b.mp3",
            ),
            "https://hf.example/org/name/resolve/main/data/a%20b.mp3"
        );
        assert_eq!(
            huggingface_file_url(
                "https://hf.example",
                RepoKind::Dataset,
                "org/name",
                "main",
                "data/a b.mp3",
            ),
            "https://hf.example/datasets/org/name/resolve/main/data/a%20b.mp3"
        );
    }

    #[test]
    fn modelscope_file_url_targets_one_file() {
        let url = modelscope_file_url(
            MS_OFFICIAL,
            RepoKind::Dataset,
            "org/name",
            "master",
            "data/a.mp3",
        );
        assert_eq!(
            url,
            "https://modelscope.cn/api/v1/datasets/org/name/repo?Revision=master&FilePath=data%2Fa.mp3"
        );
    }

    #[test]
    fn default_revisions_differ_per_backend() {
        assert_eq!(Backend::HuggingFace.default_revision(), "main");
        assert_eq!(Backend::ModelScope.default_revision(), "master");
    }

    #[test]
    fn validate_repo_value_rejects_traversal_and_absolute_paths() {
        assert!(validate_repo_value("file path", "data/a.mp3").is_ok());
        assert!(validate_repo_value("revision", "refs/pr/1").is_ok());
        assert!(validate_repo_value("file path", "file..txt").is_ok());
        assert!(validate_repo_value("file path", "./data/a.mp3").is_ok());
        assert!(validate_repo_value("file path", "../evil").is_err());
        assert!(validate_repo_value("file path", "a/../../b").is_err());
        assert!(validate_repo_value("file path", "/etc/passwd").is_err());
        assert!(validate_repo_value("revision", "..").is_err());
        assert!(validate_repo_value("revision", "/main").is_err());
        assert!(validate_repo_value("file path", "").is_err());
    }
}
