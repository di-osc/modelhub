//! Hugging Face Hub model downloading.

use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const MIRROR_ENDPOINT: &str = "https://hf-mirror.com";
const DEFAULT_CONCURRENCY: usize = 4;
const BAR_STYLE: &str = "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {decimal_bytes}/{decimal_total_bytes} • {decimal_bytes_per_sec} • {eta}";
const SPINNER_STYLE: &str = "{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}";

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Deserialize)]
struct LfsInfo {
    #[serde(default)]
    size: Option<u64>,
}

impl Sibling {
    fn expected_size(&self) -> Option<u64> {
        self.size.or_else(|| self.lfs.as_ref()?.size)
    }
}

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

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn endpoints() -> Vec<String> {
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        return vec![endpoint.trim_end_matches('/').to_owned()];
    }
    vec![MIRROR_ENDPOINT.to_owned(), DEFAULT_ENDPOINT.to_owned()]
}

fn model_info_url(endpoint: &str, model_id: &str, revision: Option<&str>) -> String {
    revision.map_or_else(
        || format!("{endpoint}/api/models/{}/", encode_path(model_id)),
        |revision| {
            format!(
                "{endpoint}/api/models/{}/revision/{}",
                encode_path(model_id),
                encode_path(revision)
            )
        },
    )
}

fn repo_dir(cache_root: &Path, model_id: &str) -> PathBuf {
    cache_root.join("models").join(model_id.replace('/', "--"))
}

fn safe_repo_path(root: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!(
            "Hugging Face file path must be relative: {}",
            path.display()
        );
    }
    let mut out = root.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!(
                    "Hugging Face file path escapes cache root: {}",
                    path.display()
                );
            }
        }
    }
    Ok(out)
}

fn safe_revision_path(root: &Path, revision: &str) -> anyhow::Result<PathBuf> {
    if revision.is_empty() {
        bail!("Hugging Face revision must not be empty");
    }
    safe_repo_path(root, revision)
}

fn partial_path(file_path: &Path) -> PathBuf {
    let name = file_path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("download");
    file_path.with_file_name(format!("{name}.part"))
}

fn complete(path: &Path, expected: Option<u64>) -> bool {
    match expected {
        Some(size) => fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() == size),
        None => fs::metadata(path).is_ok_and(|m| m.is_file()),
    }
}

fn repo_progress(
    model_id: &str,
    file_count: usize,
    total_size: Option<u64>,
) -> anyhow::Result<ProgressBar> {
    let bar = if let Some(size) = total_size.filter(|size| *size > 0) {
        let bar = ProgressBar::new(size);
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
        compact_label(model_id, 36)
    ));
    Ok(bar)
}

fn native_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Check whether a model repository is available on Hugging Face (or its mirror).
pub async fn model_exists(model_id: &str) -> anyhow::Result<bool> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let mut errors = Vec::new();
    for endpoint in endpoints() {
        let url = model_info_url(&endpoint, model_id, None);
        match auth_request(client.get(url)).send().await {
            Ok(response) if response.status().is_success() => return Ok(true),
            Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND => {
                errors.push(format!("{endpoint}: HTTP 404"));
            }
            Ok(response) => errors.push(format!("{endpoint}: HTTP {}", response.status())),
            Err(error) => errors.push(format!("{endpoint}: {error}")),
        }
    }
    if errors.iter().all(|error| error.ends_with("HTTP 404")) {
        Ok(false)
    } else {
        bail!("Hugging Face endpoints failed: {}", errors.join("; "))
    }
}

/// Return the cache directory used by the Hugging Face Hub client.
#[must_use]
pub fn cache_dir() -> PathBuf {
    native_cache_dir()
}

fn auth_request(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN")) {
        Ok(token) => builder.bearer_auth(token),
        Err(_) => builder,
    }
}

async fn download_file(
    client: Arc<reqwest::Client>,
    url: String,
    target: &Path,
    name: &str,
    expected: Option<u64>,
    progress: ProgressBar,
) -> anyhow::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if complete(target, expected) {
        if let Ok(metadata) = fs::metadata(target) {
            progress.inc(metadata.len());
        }
        return Ok(());
    }
    let part = partial_path(target);
    if part.exists() {
        fs::remove_file(&part)?;
    }

    let response = auth_request(client.get(url)).send().await?;
    if !response.status().is_success() {
        bail!(
            "failed to download Hugging Face file {name}: HTTP {}",
            response.status()
        );
    }
    let size = expected.or_else(|| response.content_length());
    let mut written = 0u64;
    {
        let mut file = BufWriter::new(fs::File::create(&part)?);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            written += chunk.len() as u64;
            file.write_all(&chunk)?;
            progress.inc(chunk.len() as u64);
        }
        file.flush()?;
    }
    if let Some(size) = size
        && written != size
    {
        let _ = fs::remove_file(&part);
        bail!("incomplete Hugging Face download for {name}: expected {size}, got {written}");
    }
    fs::rename(part, target)
        .with_context(|| format!("failed to finalize Hugging Face file {}", target.display()))?;
    Ok(())
}

/// Download a model into a modelhub-owned cache, preserving the native Hub layout.
pub async fn download_model(
    model_id: &str,
    revision: Option<&str>,
    modelhub_cache: &Path,
) -> anyhow::Result<PathBuf> {
    download_model_with_concurrency(model_id, revision, modelhub_cache, DEFAULT_CONCURRENCY).await
}

/// Download a model with a maximum number of concurrent file transfers.
pub async fn download_model_with_concurrency(
    model_id: &str,
    revision: Option<&str>,
    modelhub_cache: &Path,
    concurrency: usize,
) -> anyhow::Result<PathBuf> {
    if concurrency == 0 {
        bail!("download concurrency must be at least 1");
    }
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let mut selected_endpoint = None;
    let mut info = None;
    let mut errors = Vec::new();
    for endpoint in endpoints() {
        let url = model_info_url(&endpoint, model_id, revision);
        match auth_request(client.get(url)).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<ModelInfo>().await {
                    Ok(value) => {
                        selected_endpoint = Some(endpoint);
                        info = Some(value);
                        break;
                    }
                    Err(error) => errors.push(format!("{endpoint}: {error}")),
                }
            }
            Ok(response) => errors.push(format!("{endpoint}: HTTP {}", response.status())),
            Err(error) => errors.push(format!("{endpoint}: {error}")),
        }
    }
    let (Some(endpoint), Some(info)) = (selected_endpoint, info) else {
        bail!("Hugging Face endpoints failed: {}", errors.join("; "));
    };
    let requested = revision.unwrap_or("main");
    let snapshot_revision = info.sha.unwrap_or_else(|| requested.to_owned());
    let root = repo_dir(modelhub_cache, model_id);
    let snapshot = safe_revision_path(&root.join("snapshots"), &snapshot_revision)?;
    fs::create_dir_all(&snapshot)?;
    let file_count = info.siblings.len();
    let total_size = info
        .siblings
        .iter()
        .try_fold(0u64, |total, file| Some(total + file.expected_size()?));
    let progress = repo_progress(model_id, file_count, total_size)?;
    let progress_label = compact_label(model_id, 36);
    let client = Arc::new(client);
    let mut downloads = futures_util::stream::iter(info.siblings.into_iter().map(|sibling| {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let model_id = model_id.to_owned();
        let snapshot_revision = snapshot_revision.clone();
        let snapshot = snapshot.clone();
        let progress = progress.clone();
        async move {
            let target = safe_repo_path(&snapshot, &sibling.rfilename)?;
            let expected = sibling.expected_size();
            let url = format!(
                "{endpoint}/{}/resolve/{}/{}",
                encode_path(&model_id),
                encode_path(&snapshot_revision),
                encode_path(&sibling.rfilename)
            );
            download_file(client, url, &target, &sibling.rfilename, expected, progress).await
        }
    }))
    .buffer_unordered(concurrency);
    let mut completed = 0usize;
    while let Some(result) = downloads.next().await {
        if let Err(error) = result {
            progress.abandon_with_message(format!("{model_id} • download failed"));
            return Err(error);
        }
        completed += 1;
        progress.set_message(format!("{progress_label} • {completed}/{file_count} files"));
    }
    drop(downloads);
    progress.finish_with_message(format!("✓ {model_id} • {file_count} files"));
    let refs = root.join("refs");
    let ref_path = safe_revision_path(&refs, requested)?;
    if let Some(parent) = ref_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(ref_path, snapshot_revision)?;
    Ok(root)
}
