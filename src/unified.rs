//! Verified, content-addressed downloads shared across model backends.

use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

mod transfer;
use transfer::Transfer;

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

    /// Label used in user-facing messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
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
        encode_path(repo_id),
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
        let response = match hf_auth(client.get(url)).send().await {
            Ok(response) => response,
            // No answer at all: the endpoint is unreachable, so the next one is
            // worth trying.
            Err(error) => {
                errors.push(format!("{endpoint}: {error}"));
                continue;
            }
        };
        // The mirror serves the whole Hub, so any HTTP status it returns is
        // final; the official endpoint would add nothing.
        if !response.status().is_success() {
            errors.push(format!("{endpoint}: HTTP {}", response.status()));
            break;
        }
        let info = response.json::<HfInfo>().await?;
        if let Err(error) = validate_repo_value("revision", &info.sha) {
            errors.push(format!("{endpoint}: {error}"));
            break;
        }
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
                "{}/api/v1/models/{}/repo/files?Recursive=true&Revision={}",
                ms_base_url(),
                encode_path(repo_id),
                urlencoding::encode(revision)
            );
            let response = client.get(url).send().await?;
            modelscope_files_from_response(response).await
        }
        RepoKind::Dataset => {
            let mut all = Vec::new();
            for page in 1usize.. {
                let url = format!(
                    "{}/api/v1/datasets/{}/repo/tree?Recursive=True&Revision={}&PageNumber={page}&PageSize={DATASET_PAGE_SIZE}",
                    ms_base_url(),
                    encode_path(repo_id),
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
    let (mut sha256, mut git) = hashers(size);
    hash_contents(fs::File::open(path)?, &mut sha256, &mut git)?;
    Ok((
        format!("{:x}", sha256.finalize()),
        format!("{:x}", git.finalize()),
    ))
}

fn hashers(size: u64) -> (Sha256, Sha1) {
    let mut git = Sha1::new();
    git.update(format!("blob {size}\0").as_bytes());
    (Sha256::new(), git)
}

fn hash_contents(mut reader: impl Read, sha256: &mut Sha256, git: &mut Sha1) -> anyhow::Result<()> {
    let mut buffer = vec![0u8; 64 * 1024].into_boxed_slice();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        sha256.update(&buffer[..count]);
        git.update(&buffer[..count]);
    }
    Ok(())
}

/// Name the backend a file is actually fetched from, so the bar reflects the
/// live source rather than every backend that hosts the repository.
///
/// The backend leads the message so a long repository id or path can only push
/// the file name into the truncated tail, never the backend itself.
fn announce_source(progress: &ProgressBar, repo_id: &str, remote: &RemoteFile) {
    crate::progress::set_message(
        progress,
        format!("{} • {repo_id} • {}", remote.backend.label(), remote.path),
    );
}

async fn materialize(
    remote: &RemoteFile,
    repo_id: &str,
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
    announce_source(progress, repo_id, remote);
    let request = file_request(remote.backend, &remote.url, hf_client, ms_client);
    Transfer::start(
        request,
        remote.backend,
        &remote.url,
        cache_root,
        Some(remote),
    )
    .await
    .with_context(|| format!("failed to download {}", remote.path))?
    .finish(&remote.path, cache_root, progress)
    .await
}

fn file_request(
    backend: Backend,
    url: &str,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
) -> reqwest::RequestBuilder {
    match backend {
        Backend::HuggingFace => hf_auth(hf_client.get(url)),
        Backend::ModelScope => ms_client.get(url).header(
            crate::modelscope::client::USER_AGENT.0,
            crate::modelscope::client::USER_AGENT.1,
        ),
    }
}

fn link_artifact(blob: &Path, target: &Path) -> anyhow::Result<()> {
    if target.exists() {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::hard_link(blob, target) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(_) => fs::copy(blob, target)
            .map(|_| ())
            .with_context(|| format!("failed to materialize {}", target.display())),
    }
}

/// Bytes a repository download is expected to transfer: the files resolved
/// across backends (downloaded once and compared) plus the per-backend
/// artifacts. Cache hits still count, so the bar reflects every planned byte.
fn download_total(git_comparisons: &[(RemoteFile, RemoteFile)], plans: &[Artifact]) -> u64 {
    let compared = git_comparisons
        .iter()
        .map(|(_, modelscope)| modelscope.size)
        .sum::<u64>();
    let planned = plans
        .iter()
        .map(|artifact| artifact.remote.size)
        .sum::<u64>();
    compared.saturating_add(planned)
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
    cache_root: &Path,
) -> anyhow::Result<(RepoKind, Backend, Transfer<'static>)> {
    let attempts = candidates.len().max(1);
    let requests = futures_util::stream::iter(candidates.into_iter().map(
        |(kind, backend, url)| async move {
            let request = file_request(backend, &url, hf_client, ms_client);
            let transfer = Transfer::start(request, backend, &url, cache_root, None)
                .await
                .with_context(|| url)?;
            Ok::<_, anyhow::Error>((kind, backend, transfer))
        },
    ))
    .buffer_unordered(attempts);
    futures_util::pin_mut!(requests);
    let mut errors = Vec::new();
    while let Some(result) = requests.next().await {
        match result {
            Ok(winner) => return Ok(winner),
            Err(error) => errors.push(format!("{error:#}")),
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
/// without touching the network. A ref whose value is not a safe relative path
/// is ignored.
fn snapshot_revisions(repo_root: &Path, backend: Backend, revision: &str) -> Vec<String> {
    let mut revisions = vec![revision.to_owned()];
    if backend == Backend::HuggingFace
        && let Ok(sha) =
            fs::read_to_string(repo_root.join("huggingface").join("refs").join(revision))
    {
        let sha = sha.trim();
        if !sha.is_empty()
            && sha != revision
            && validate_repo_value("snapshot revision", sha).is_ok()
        {
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
    let bar = crate::progress::bar(progress, format!("{repo_id} • {file}"), None);
    let outcome = async {
        let (kind, backend, transfer) = send_first_success(
            candidates, file, repo_id, &hf_client, &ms_client, cache_root,
        )
        .await?;
        crate::progress::set_message(&bar, format!("{} • {repo_id} • {file}", backend.label()));
        if let Some(length) = transfer.size().filter(|length| *length > 0) {
            bar.set_style(crate::progress::bar_style());
            bar.set_length(length);
        }
        let (blob, _, _) = transfer.finish(file, cache_root, &bar).await?;
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
/// `kind`/`backend` narrow the detection: a known kind skips probing the other
/// kind, a known backend skips probing the other backend. `progress` toggles the
/// progress bar.
#[allow(clippy::too_many_arguments)]
pub async fn download_repo(
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
    cache_root: &Path,
    concurrency: usize,
    allow_weight_mismatch: bool,
    progress: bool,
    kind: Option<RepoKind>,
    backend: Option<Backend>,
) -> anyhow::Result<DownloadedRepo> {
    validate_repo_value("repo id", repo_id)?;
    validate_repo_value("revision", huggingface_revision)?;
    validate_repo_value("revision", modelscope_revision)?;
    let hf_client = huggingface_client()?;
    let ms_client = crate::modelscope::client::http_client()?;
    let (kind, hf, ms) = detect_manifests(
        &hf_client,
        &ms_client,
        repo_id,
        huggingface_revision,
        modelscope_revision,
        kind,
        backend,
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
        progress,
    )
    .await
}

fn huggingface_client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?)
}

/// Quietly probe the selected manifests to decide what `repo_id` is.
///
/// A repository that exists as both a model and a dataset is rejected so the
/// caller does not have to guess which one was meant. `want_kind`/`want_backend`
/// skip the corresponding probes entirely.
async fn detect_manifests(
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
    want_kind: Option<RepoKind>,
    want_backend: Option<Backend>,
) -> anyhow::Result<(RepoKind, Option<Manifest>, Option<Manifest>)> {
    // A model hit is final, so the dataset is probed only when the model probe
    // misses (or the kind was hinted). This keeps a model download from waiting
    // on a dataset probe that cannot change the outcome.
    if want_kind.is_none_or(|kind| kind == RepoKind::Model) {
        let (hf, ms) = probe_manifests(
            RepoKind::Model,
            want_backend,
            hf_client,
            ms_client,
            repo_id,
            huggingface_revision,
            modelscope_revision,
        )
        .await;
        if hf.is_some() || ms.is_some() {
            return Ok((RepoKind::Model, hf, ms));
        }
    }
    if want_kind.is_none_or(|kind| kind == RepoKind::Dataset) {
        let (hf, ms) = probe_manifests(
            RepoKind::Dataset,
            want_backend,
            hf_client,
            ms_client,
            repo_id,
            huggingface_revision,
            modelscope_revision,
        )
        .await;
        if hf.is_some() || ms.is_some() {
            return Ok((RepoKind::Dataset, hf, ms));
        }
    }
    bail!(
        "`{repo_id}` was not found as a model or a dataset on any supported backend; \
         check the ID, revision, and credentials (HF_TOKEN for Hugging Face)"
    );
}

/// Probe one kind on the selected backends, discarding error details.
async fn probe_manifests(
    kind: RepoKind,
    backend: Option<Backend>,
    hf_client: &reqwest::Client,
    ms_client: &reqwest::Client,
    repo_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
) -> (Option<Manifest>, Option<Manifest>) {
    let wanted = |candidate: Backend| backend.is_none_or(|want| want == candidate);
    let hf = async {
        if wanted(Backend::HuggingFace) {
            huggingface_manifest(hf_client, kind, repo_id, huggingface_revision)
                .await
                .ok()
        } else {
            None
        }
    };
    let ms = async {
        if wanted(Backend::ModelScope) {
            modelscope_manifest(ms_client, kind, repo_id, modelscope_revision)
                .await
                .ok()
        } else {
            None
        }
    };
    tokio::join!(hf, ms)
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
    progress: bool,
) -> anyhow::Result<DownloadedRepo> {
    if concurrency == 0 {
        bail!("download concurrency must be at least 1");
    }
    let repo_root = cache_root
        .join(kind.segment())
        .join(repo_id.replace('/', "--"));
    ensure_repo_markers(&repo_root, repo_id)?;
    let mut paths = BTreeSet::new();
    if let Some(manifest) = hf.as_ref() {
        paths.extend(manifest.files.keys().cloned());
    }
    if let Some(manifest) = ms.as_ref() {
        paths.extend(manifest.files.keys().cloned());
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
    // A repository whose manifest lists no files still needs its backend root to
    // exist, otherwise `ops::download` cannot link the empty snapshot into the
    // native backend cache.
    if let Some(root) = &hf_root {
        fs::create_dir_all(root)?;
    }
    if let Some(root) = &ms_root {
        fs::create_dir_all(root)?;
    }
    let hf_snapshot_revision = hf.as_ref().map(|manifest| manifest.revision.clone());
    let ms_snapshot_revision = ms.as_ref().map(|manifest| manifest.revision.clone());
    let total = download_total(&git_comparisons, &plans);
    // The source backend is filled in per file by `announce_source`.
    let progress = crate::progress::bar(progress, repo_id.to_owned(), Some(total));
    for (hf_file, ms_file) in git_comparisons {
        let (blob, sha256, git) = materialize(
            &ms_file, repo_id, cache_root, hf_client, ms_client, &progress,
        )
        .await?;
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
            progress.inc_length(hf_file.size);
            add_separate(&mut plans, hf_file);
        }
    }
    let hf_client = Arc::new(hf_client.clone());
    let ms_client = Arc::new(ms_client.clone());
    let downloads = futures_util::stream::iter(plans.into_iter().map(|artifact| {
        let cache_root = cache_root.to_path_buf();
        let repo_root = repo_root.clone();
        let hf_client = hf_client.clone();
        let ms_client = ms_client.clone();
        let progress = progress.clone();
        let repo_id = repo_id.to_owned();
        let hf_revision = hf_snapshot_revision.clone();
        let ms_revision = ms_snapshot_revision.clone();
        async move {
            let (blob, _, _) = materialize(
                &artifact.remote,
                &repo_id,
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
        let reference = repo_root
            .join("huggingface")
            .join("refs")
            .join(huggingface_revision);
        if let Some(parent) = reference.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(reference, &manifest.revision)?;
    }
    write_repo_manifest(&repo_root, kind, hf.as_ref(), ms.as_ref())?;
    Ok(DownloadedRepo {
        kind,
        repo_root: repo_root.clone(),
        huggingface_root: hf_root,
        modelscope_root: ms_root,
        file: None,
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
    fn backend_labels_are_stable() {
        assert_eq!(Backend::HuggingFace.label(), "huggingface");
        assert_eq!(Backend::ModelScope.label(), "modelscope");
    }

    #[test]
    fn download_total_counts_compared_and_planned_files() {
        let remote = |size| RemoteFile {
            backend: Backend::HuggingFace,
            path: "file".to_owned(),
            size,
            sha256: None,
            git_blob_id: None,
            url: String::new(),
        };
        // The comparison downloads the ModelScope copy once, then links it to
        // Hugging Face, so only the ModelScope size counts.
        let comparisons = [(remote(100), remote(200))];
        let plans = [
            Artifact {
                remote: remote(300),
                targets: vec![(Backend::HuggingFace, "file".to_owned())],
            },
            Artifact {
                remote: remote(400),
                targets: vec![(Backend::ModelScope, "file".to_owned())],
            },
        ];
        assert_eq!(download_total(&comparisons, &plans), 900);
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
