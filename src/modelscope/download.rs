//! Vendored `ModelScope` downloader.
//!
//! Kept in-tree so we control the download UX instead of relying on the
//! external `modelscope` crate.

use super::client::{USER_AGENT, http_client};
use super::types::{ModelScopeResponse, RepoFile};
use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const MODEL_FILES_URL: &str =
    "https://modelscope.cn/api/v1/models/<repo_id>/repo/files?Recursive=true&Revision=<revision>";
const DATASET_FILES_URL: &str = "https://modelscope.cn/api/v1/datasets/<repo_id>/repo/tree?Recursive=True&Revision=<revision>&PageNumber=<page>&PageSize=<page_size>";
const FILE_DOWNLOAD_URL: &str =
    "https://modelscope.cn/api/v1/<segment>/<repo_id>/repo?Revision=<revision>&FilePath=<path>";
const BAR_STYLE: &str = "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {decimal_bytes}/{decimal_total_bytes} • {decimal_bytes_per_sec} • {eta}";
const SPINNER_STYLE: &str = "{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}";
const DEFAULT_REVISION: &str = "master";
const DEFAULT_CONCURRENCY: usize = 4;
const DATASET_PAGE_SIZE: usize = 200;

fn compact_label(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let compact: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!(
            "{}…",
            compact.chars().take(max_chars - 1).collect::<String>()
        )
    } else {
        compact
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RepoKind {
    Model,
    Dataset,
}

impl RepoKind {
    const fn segment(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Dataset => "dataset",
        }
    }
}

fn repo_cache_dir(cache_root: &Path, kind: RepoKind, repo_id: &str, revision: &str) -> PathBuf {
    cache_root
        .join(kind.segment())
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join(revision)
}

fn validate_revision(revision: &str) -> anyhow::Result<()> {
    let path = Path::new(revision);
    let mut components = path.components();
    if revision.is_empty()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        bail!("revision must be a single path component: {revision}");
    }
    Ok(())
}

fn partial_path_for(file_path: &Path) -> PathBuf {
    let name = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    file_path.with_file_name(format!("{name}.part"))
}

fn file_is_complete(file_path: &Path, size: u64) -> bool {
    size > 0 && fs::metadata(file_path).is_ok_and(|meta| meta.len() == size)
}

fn safe_repo_path(root: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!("repository file path must be relative: {}", path.display());
    }

    let mut out = root.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!(
                    "repository file path escapes the cache root: {}",
                    path.display()
                );
            }
        }
    }

    Ok(out)
}

fn download_url(kind: RepoKind, repo_id: &str, revision: &str, path: &str) -> String {
    FILE_DOWNLOAD_URL
        .replace("<segment>", kind.segment())
        .replace("<repo_id>", repo_id)
        .replace("<revision>", &urlencoding::encode(revision))
        .replace("<path>", &urlencoding::encode(path))
}

fn progress_bar(file_name: &str, size: Option<u64>) -> anyhow::Result<ProgressBar> {
    let bar = ProgressBar::new(size.unwrap_or(0));
    let style = ProgressStyle::default_bar().template(BAR_STYLE)?;
    bar.set_style(style);
    bar.set_message(compact_label(file_name, 32));
    if let Some(size) = size {
        bar.set_length(size);
    }
    Ok(bar)
}

fn repo_progress(repo_id: &str, file_count: usize, total_size: u64) -> anyhow::Result<ProgressBar> {
    let bar = if total_size > 0 {
        let bar = ProgressBar::new(total_size);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(BAR_STYLE)?
                .progress_chars("━━─"),
        );
        bar
    } else {
        let bar = ProgressBar::new_spinner();
        bar.set_style(ProgressStyle::default_spinner().template(SPINNER_STYLE)?);
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        bar
    };
    bar.set_message(format!(
        "{} • 0/{file_count} files",
        compact_label(repo_id, 36)
    ));
    Ok(bar)
}

async fn download_to_path(
    client: Arc<reqwest::Client>,
    url: &str,
    file_path: &Path,
    file_name: &str,
    expected_size: Option<u64>,
    bar: ProgressBar,
    manage_bar: bool,
) -> anyhow::Result<PathBuf> {
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    if let Some(size) = expected_size
        && file_is_complete(file_path, size)
    {
        if manage_bar {
            bar.finish_and_clear();
        }
        return Ok(file_path.to_path_buf());
    }

    let part_path = partial_path_for(file_path);
    if part_path.exists() {
        fs::remove_file(&part_path).with_context(|| {
            format!(
                "failed to remove stale partial file {}",
                part_path.display()
            )
        })?;
    }

    let response = client
        .get(url)
        .header(USER_AGENT.0, USER_AGENT.1)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        if manage_bar {
            bar.abandon();
        }
        bail!("failed to download file {file_name}: HTTP {status}");
    }

    let size = expected_size.or_else(|| response.content_length());
    if let Some(size) = size
        && manage_bar
    {
        bar.set_length(size);
    }

    let mut written = 0u64;
    {
        let mut file = BufWriter::new(fs::File::create(&part_path)?);
        let mut stream = response.bytes_stream();
        while let Some(item) = stream.next().await {
            let chunk = item?;
            file.write_all(&chunk)?;
            written += chunk.len() as u64;
            bar.inc(chunk.len() as u64);
        }
        file.flush()?;
    }

    if let Some(size) = size
        && written != size
    {
        let _ = fs::remove_file(&part_path);
        if manage_bar {
            bar.abandon();
        }
        bail!("incomplete download for {file_name}: expected {size} bytes, got {written} bytes");
    }

    fs::rename(&part_path, file_path).with_context(|| {
        format!(
            "failed to move {} to {}",
            part_path.display(),
            file_path.display()
        )
    })?;
    if manage_bar {
        bar.finish_and_clear();
    }
    Ok(file_path.to_path_buf())
}

async fn download_repo_file(
    client: Arc<reqwest::Client>,
    kind: RepoKind,
    repo_id: String,
    revision: String,
    repo_file: RepoFile,
    snapshot_dir: PathBuf,
    bar: ProgressBar,
) -> anyhow::Result<PathBuf> {
    let path = repo_file.path;
    let file_path = safe_repo_path(&snapshot_dir, &path)?;
    let url = download_url(kind, &repo_id, &revision, &path);
    let file_name = repo_file.name;
    let expected_size = (repo_file.size > 0).then_some(repo_file.size);
    download_to_path(
        client,
        &url,
        &file_path,
        &file_name,
        expected_size,
        bar,
        false,
    )
    .await
}

async fn list_repo_files(
    client: &reqwest::Client,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    match kind {
        RepoKind::Model => list_model_files(client, repo_id, revision).await,
        RepoKind::Dataset => list_dataset_files(client, repo_id, revision).await,
    }
}

async fn list_model_files(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    let files_url = MODEL_FILES_URL
        .replace("<repo_id>", repo_id)
        .replace("<revision>", &urlencoding::encode(revision));

    let response = fetch_file_list(client, &files_url, RepoKind::Model, repo_id, revision).await?;
    Ok(response
        .data
        .context("ModelScope response did not include file data")?
        .files)
}

/// Check whether a model repository is available on `ModelScope` without downloading it.
pub async fn model_exists(model_id: &str, revision: &str) -> anyhow::Result<bool> {
    validate_revision(revision)?;
    let client = http_client().await?;
    list_model_files(&client, model_id, revision)
        .await
        .map(|_| true)
}

async fn list_dataset_files(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    let mut all_files = Vec::new();

    for page in 1usize.. {
        let files_url = DATASET_FILES_URL
            .replace("<repo_id>", repo_id)
            .replace("<revision>", &urlencoding::encode(revision))
            .replace("<page>", &page.to_string())
            .replace("<page_size>", &DATASET_PAGE_SIZE.to_string());

        let response =
            fetch_file_list(client, &files_url, RepoKind::Dataset, repo_id, revision).await?;
        let files = response
            .data
            .context("ModelScope response did not include file data")?
            .files;
        let count = files.len();
        all_files.extend(files);

        if count < DATASET_PAGE_SIZE {
            break;
        }
    }

    Ok(all_files)
}

async fn fetch_file_list(
    client: &reqwest::Client,
    files_url: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<ModelScopeResponse> {
    let resp = client.get(files_url).send().await?;
    if !resp.status().is_success() {
        bail!(
            "Failed to list {} files for {repo_id}@{revision}: {}\nTip: Maybe the ID, revision, or login state is incorrect",
            kind.label(),
            resp.text().await?
        );
    }

    let response = resp.json::<ModelScopeResponse>().await?;
    if !response.success {
        bail!(
            "Failed to list {} files: {}",
            kind.label(),
            response.message
        );
    }

    Ok(response)
}

async fn download_repo_snapshot(
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
    concurrency: usize,
) -> anyhow::Result<PathBuf> {
    validate_revision(revision)?;
    if concurrency == 0 {
        bail!("download concurrency must be at least 1");
    }
    let save_dir = save_dir.into();
    fs::create_dir_all(&save_dir)?;

    let snapshot_dir = repo_cache_dir(&save_dir, kind, repo_id, revision);
    fs::create_dir_all(&snapshot_dir)?;

    let client = Arc::new(http_client().await?);
    let repo_files: Vec<_> = list_repo_files(&client, kind, repo_id, revision)
        .await?
        .into_iter()
        .filter(|f| f.file_type != "tree")
        .filter(|f| {
            let Ok(file_path) = safe_repo_path(&snapshot_dir, &f.path) else {
                return true;
            };
            !file_is_complete(&file_path, f.size)
        })
        .collect();

    if repo_files.is_empty() {
        tracing::info!(
            "{} `{repo_id}` cache is complete at {}",
            kind.label(),
            snapshot_dir.display()
        );
        return Ok(snapshot_dir);
    }

    let file_count = repo_files.len();
    let total_size = repo_files.iter().map(|file| file.size).sum();
    let progress = repo_progress(repo_id, file_count, total_size)?;
    let progress_label = compact_label(repo_id, 36);
    let mut downloads = futures_util::stream::iter(repo_files.into_iter().map(|repo_file| {
        let client = client.clone();
        let repo_id = repo_id.to_string();
        let revision = revision.to_string();
        let snapshot_dir = snapshot_dir.clone();
        let progress = progress.clone();
        async move {
            download_repo_file(
                client,
                kind,
                repo_id,
                revision,
                repo_file,
                snapshot_dir,
                progress,
            )
            .await
        }
    }))
    .buffer_unordered(concurrency);
    let mut completed = 0usize;
    while let Some(result) = downloads.next().await {
        if let Err(error) = result {
            progress.abandon_with_message(format!("{repo_id} • download failed"));
            return Err(error);
        }
        completed += 1;
        progress.set_message(format!("{progress_label} • {completed}/{file_count} files"));
    }
    drop(downloads);
    progress.finish_with_message(format!("✓ {repo_id} • {file_count} files"));

    tracing::info!(
        "Downloaded {} `{repo_id}@{revision}` to {}",
        kind.label(),
        snapshot_dir.display()
    );
    Ok(snapshot_dir)
}

async fn download_single_file(
    kind: RepoKind,
    repo_id: &str,
    file_path: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<PathBuf> {
    validate_revision(revision)?;
    let save_dir = save_dir.into();
    let snapshot_dir = repo_cache_dir(&save_dir, kind, repo_id, revision);
    let target = safe_repo_path(&snapshot_dir, file_path)?;
    let file_name = Path::new(file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(file_path);

    if target.exists() {
        return Ok(target);
    }

    let client = Arc::new(http_client().await?);
    let url = download_url(kind, repo_id, revision, file_path);
    let bar = progress_bar(file_name, None)?;
    download_to_path(client, &url, &target, file_name, None, bar, true).await
}

/// Download a model from `ModelScope` into `save_dir`.
///
/// Uses the `master` revision.
pub async fn download_model(model_id: &str, save_dir: impl Into<PathBuf>) -> anyhow::Result<()> {
    download_model_revision(model_id, DEFAULT_REVISION, save_dir).await
}

/// Download a model revision from `ModelScope` into `save_dir`.
///
/// `save_dir` is the `ModelScope` cache root (e.g. `$HOME/.cache/modelscope`).
/// The actual files are placed under
/// `save_dir/models/<namespace--model>/snapshots/<revision>`.
///
/// Progress bars are displayed only when files actually need downloading.
pub async fn download_model_revision(
    model_id: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<()> {
    download_model_revision_with_concurrency(model_id, revision, save_dir, DEFAULT_CONCURRENCY)
        .await
}

/// Download a model revision with a maximum number of concurrent file transfers.
pub async fn download_model_revision_with_concurrency(
    model_id: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
    concurrency: usize,
) -> anyhow::Result<()> {
    download_repo_snapshot(RepoKind::Model, model_id, revision, save_dir, concurrency)
        .await
        .map(|_| ())
}

/// Download a dataset from `ModelScope` into `save_dir`.
///
/// Uses the `master` revision.
pub async fn download_dataset(
    dataset_id: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<()> {
    download_dataset_revision(dataset_id, DEFAULT_REVISION, save_dir).await
}

/// Download a dataset revision from `ModelScope` into `save_dir`.
///
/// `save_dir` is the `ModelScope` cache root (e.g. `$HOME/.cache/modelscope`).
/// The actual files are placed under
/// `save_dir/datasets/<namespace--dataset>/snapshots/<revision>`.
pub async fn download_dataset_revision(
    dataset_id: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<()> {
    download_repo_snapshot(
        RepoKind::Dataset,
        dataset_id,
        revision,
        save_dir,
        DEFAULT_CONCURRENCY,
    )
    .await
    .map(|_| ())
}

/// Download a single model file from the `master` revision.
pub async fn download_model_file(
    model_id: &str,
    file_path: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<PathBuf> {
    download_model_file_revision(model_id, file_path, DEFAULT_REVISION, save_dir).await
}

/// Download a single model file from a specific revision.
pub async fn download_model_file_revision(
    model_id: &str,
    file_path: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<PathBuf> {
    download_single_file(RepoKind::Model, model_id, file_path, revision, save_dir).await
}

/// Download a single dataset file from the `master` revision.
pub async fn download_dataset_file(
    dataset_id: &str,
    file_path: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<PathBuf> {
    download_dataset_file_revision(dataset_id, file_path, DEFAULT_REVISION, save_dir).await
}

/// Download a single dataset file from a specific revision.
pub async fn download_dataset_file_revision(
    dataset_id: &str,
    file_path: &str,
    revision: &str,
    save_dir: impl Into<PathBuf>,
) -> anyhow::Result<PathBuf> {
    download_single_file(RepoKind::Dataset, dataset_id, file_path, revision, save_dir).await
}
