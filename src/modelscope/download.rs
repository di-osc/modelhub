//! Vendored `ModelScope` downloader.
//!
//! Kept in-tree so we control the download UX instead of relying on the
//! external `modelscope` crate.

use super::client::{USER_AGENT, http_client};
use super::types::{ModelScopeResponse, RepoFile};
use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FILES_URL: &str = "https://modelscope.cn/api/v1/models/<model_id>/repo/files?Recursive=true";
const DOWNLOAD_URL: &str = "https://modelscope.cn/models/<model_id>/resolve/master/<path>";
const BAR_STYLE: &str = "{msg:<30} {bar} {decimal_bytes:<10} / {decimal_total_bytes:<10} {decimal_bytes_per_sec:<12} {percent:<3}%  {eta_precise}";

fn partial_path_for(file_path: &Path) -> PathBuf {
    let name = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    file_path.with_file_name(format!("{name}.part"))
}

fn file_is_complete(file_path: &Path, size: u64) -> bool {
    fs::metadata(file_path).is_ok_and(|meta| meta.len() == size)
}

async fn download_file(
    client: Arc<reqwest::Client>,
    model_id: String,
    repo_file: RepoFile,
    save_dir: PathBuf,
    bar: ProgressBar,
) -> anyhow::Result<()> {
    let path = &repo_file.path;
    let file_path = save_dir.join(path);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // If the final file exists with a wrong size, leave it in place while
    // downloading to a .part file and replace it only after the new download has
    // completed successfully. Complete cached files are filtered before tasks
    // are spawned, so this is only a race-safety fallback.
    if file_is_complete(&file_path, repo_file.size) {
        bar.finish_and_clear();
        return Ok(());
    }

    bar.set_message(repo_file.name.clone());
    bar.set_length(repo_file.size);

    let url = DOWNLOAD_URL
        .replace("<model_id>", &model_id)
        .replace("<path>", path);

    let part_path = partial_path_for(&file_path);
    if part_path.exists() {
        fs::remove_file(&part_path).with_context(|| {
            format!(
                "failed to remove stale partial file {}",
                part_path.display()
            )
        })?;
    }

    let response = client
        .get(&url)
        .header(USER_AGENT.0, USER_AGENT.1)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        bar.abandon();
        bail!(
            "Failed to download file {}: HTTP {}",
            repo_file.name,
            status
        );
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

    if written != repo_file.size {
        let _ = fs::remove_file(&part_path);
        bar.abandon();
        bail!(
            "incomplete download for {}: expected {} bytes, got {} bytes",
            repo_file.name,
            repo_file.size,
            written
        );
    }

    fs::rename(&part_path, &file_path).with_context(|| {
        format!(
            "failed to move {} to {}",
            part_path.display(),
            file_path.display()
        )
    })?;
    bar.finish();
    Ok(())
}

/// Download a model from `ModelScope` into `save_dir`.
///
/// `save_dir` is the *cache root* (e.g. `$HOME/.cache/vasr`).
/// The actual files are placed under `save_dir/<model_id>`.
///
/// Progress bars are displayed only when files actually need downloading.
pub async fn download_model(model_id: &str, save_dir: impl Into<PathBuf>) -> anyhow::Result<()> {
    let save_dir = save_dir.into();
    fs::create_dir_all(&save_dir)?;

    let model_dir = save_dir.join(model_id);
    fs::create_dir_all(&model_dir)?;

    let files_url = FILES_URL.replace("<model_id>", model_id);
    let client = Arc::new(http_client().await?);

    let resp = client.get(files_url).send().await?;
    if !resp.status().is_success() {
        bail!(
            "Failed to list model files for {model_id}: {}\nTip: Maybe the model ID is incorrect or login is required",
            resp.text().await?
        );
    }

    let response = resp.json::<ModelScopeResponse>().await?;
    if !response.success {
        bail!("Failed to list model files: {}", response.message);
    }

    let repo_files: Vec<_> = response
        .data
        .context("ModelScope response did not include file data")?
        .files
        .into_iter()
        .filter(|f| f.file_type == "blob")
        .filter(|f| {
            let file_path = model_dir.join(&f.path);
            !file_is_complete(&file_path, f.size)
        })
        .collect();

    if repo_files.is_empty() {
        tracing::info!(
            "Model `{model_id}` cache is complete at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let bars = MultiProgress::new();
    let mut tasks = Vec::new();

    for repo_file in repo_files {
        let bar = ProgressBar::new(0);
        let style = ProgressStyle::default_bar().template(BAR_STYLE)?;
        bar.set_style(style);
        bars.add(bar.clone());

        let client = client.clone();
        let model_id = model_id.to_string();
        let save_dir = model_dir.clone();
        let task =
            tokio::spawn(
                async move { download_file(client, model_id, repo_file, save_dir, bar).await },
            );
        tasks.push(task);
    }

    for task in tasks {
        task.await??;
    }

    tracing::info!("Downloaded `{model_id}` to {}", model_dir.display());
    Ok(())
}
