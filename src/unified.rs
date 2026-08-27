//! Verified, content-addressed downloads shared across model backends.

use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const HF_MIRROR: &str = "https://hf-mirror.com";
const HF_OFFICIAL: &str = "https://huggingface.co";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    HuggingFace,
    ModelScope,
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
pub struct DownloadedModel {
    pub model_root: PathBuf,
    pub huggingface_root: Option<PathBuf>,
    pub modelscope_root: Option<PathBuf>,
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
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Message")]
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

fn hf_auth(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN")) {
        Ok(token) => request.bearer_auth(token),
        Err(_) => request,
    }
}

async fn huggingface_manifest(
    client: &reqwest::Client,
    model_id: &str,
    revision: &str,
) -> anyhow::Result<Manifest> {
    let mut errors = Vec::new();
    for endpoint in hf_endpoints() {
        let url = format!(
            "{endpoint}/api/models/{}/revision/{}?blobs=true",
            encode_path(model_id),
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
                            url: format!(
                                "{endpoint}/{}/resolve/{}/{}",
                                encode_path(model_id),
                                encode_path(&info.sha),
                                encode_path(&path)
                            ),
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

async fn modelscope_manifest(model_id: &str, revision: &str) -> anyhow::Result<Manifest> {
    let client = crate::modelscope::client::http_client().await?;
    let url = format!(
        "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Recursive=true&Revision={}",
        urlencoding::encode(revision)
    );
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        bail!("ModelScope manifest failed: HTTP {}", response.status());
    }
    let response = response.json::<MsResponse>().await?;
    if !response.success {
        bail!("ModelScope manifest failed: {}", response.message);
    }
    let files = response
        .data
        .context("ModelScope manifest did not include data")?
        .files
        .into_iter()
        .filter(|file| file.file_type != "tree")
        .map(|file| {
            let path = file.path;
            let remote = RemoteFile {
                backend: Backend::ModelScope,
                size: file.size,
                sha256: file.sha256.filter(|hash| !hash.is_empty()),
                git_blob_id: None,
                url: format!(
                    "https://modelscope.cn/api/v1/models/{model_id}/repo?Revision={}&FilePath={}",
                    urlencoding::encode(revision),
                    urlencoding::encode(&path)
                ),
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
            progress.inc(fs::metadata(&cached)?.len());
            let (_, git) = digest_file(&cached, remote.size)?;
            return Ok((cached, hash.clone(), git));
        }
    }
    let staging_target = safe_path(
        &cache_root.join("staging").join(match remote.backend {
            Backend::HuggingFace => "huggingface",
            Backend::ModelScope => "modelscope",
        }),
        &remote.path,
    )?;
    let staging_name = staging_target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    let staging = staging_target.with_file_name(format!("{staging_name}.part"));
    if let Some(parent) = staging.parent() {
        fs::create_dir_all(parent)?;
    }
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
    let mut sha256 = Sha256::new();
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", remote.size).as_bytes());
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
    if remote.size > 0 && written != remote.size {
        let _ = fs::remove_file(&staging);
        bail!("incomplete download for {}", remote.path);
    }
    let sha256 = format!("{:x}", sha256.finalize());
    let git = format!("{:x}", git.finalize());
    if remote
        .sha256
        .as_ref()
        .is_some_and(|expected| expected != &sha256)
    {
        let _ = fs::remove_file(&staging);
        bail!("SHA-256 mismatch for {}", remote.path);
    }
    if remote
        .git_blob_id
        .as_ref()
        .is_some_and(|expected| expected != &git)
    {
        let _ = fs::remove_file(&staging);
        bail!("Git blob hash mismatch for {}", remote.path);
    }
    let blob = cache_root.join("blobs").join("sha256").join(&sha256);
    if let Some(parent) = blob.parent() {
        fs::create_dir_all(parent)?;
    }
    if blob.exists() {
        fs::remove_file(&staging)?;
    } else {
        // Another concurrent file may have produced the same content-addressed
        // blob between the existence check and the rename. In that case the
        // already-complete blob wins and this staging file can be discarded.
        if let Err(error) = fs::rename(&staging, &blob) {
            if blob.exists() {
                fs::remove_file(&staging)?;
            } else {
                return Err(error.into());
            }
        }
    }
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

/// Download backend manifests, verify hashes, and materialize deduplicated snapshots.
#[allow(clippy::too_many_lines)]
pub async fn download_model(
    model_id: &str,
    huggingface_revision: &str,
    modelscope_revision: &str,
    cache_root: &Path,
    concurrency: usize,
    allow_weight_mismatch: bool,
) -> anyhow::Result<DownloadedModel> {
    if concurrency == 0 {
        bail!("download concurrency must be at least 1");
    }
    let hf_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let ms_client = crate::modelscope::client::http_client().await?;
    let (hf_result, ms_result) = tokio::join!(
        huggingface_manifest(&hf_client, model_id, huggingface_revision),
        modelscope_manifest(model_id, modelscope_revision)
    );
    let hf = hf_result
        .map_err(|error| eprintln!("Hugging Face manifest unavailable: {error:#}"))
        .ok();
    let ms = ms_result
        .map_err(|error| eprintln!("ModelScope manifest unavailable: {error:#}"))
        .ok();
    if hf.is_none() && ms.is_none() {
        bail!("model `{model_id}` was not found on any supported backend");
    }
    let model_root = cache_root.join("models").join(model_id.replace('/', "--"));
    fs::create_dir_all(&model_root)?;
    fs::write(model_root.join(".modelhub-model-id"), model_id)?;
    fs::write(model_root.join(".modelhub-layout"), "cas-v1")?;
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
                    if is_critical(&path) && !allow_weight_mismatch {
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
    let hf_root = hf.as_ref().map(|_| model_root.join("huggingface"));
    let ms_root = ms.as_ref().map(|_| model_root.join("modelscope"));
    let hf_snapshot_revision = hf.as_ref().map(|manifest| manifest.revision.clone());
    let ms_snapshot_revision = ms.as_ref().map(|manifest| manifest.revision.clone());
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}")?,
    );
    progress.set_message(format!("{model_id} • verifying and downloading"));
    progress.enable_steady_tick(std::time::Duration::from_millis(100));
    for (hf_file, ms_file) in git_comparisons {
        let (blob, sha256, git) =
            materialize(&ms_file, cache_root, &hf_client, &ms_client, &progress).await?;
        let ms_target = safe_path(
            &snapshot_root(
                &model_root,
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
                    &model_root,
                    Backend::HuggingFace,
                    hf_snapshot_revision
                        .as_deref()
                        .context("missing Hugging Face revision")?,
                ),
                &hf_file.path,
            )?;
            link_artifact(&blob, &hf_target)?;
        } else if is_critical(&hf_file.path) && !allow_weight_mismatch {
            bail!("critical file differs across backends: {}", hf_file.path);
        } else {
            add_separate(&mut plans, hf_file);
        }
    }
    let hf_client = Arc::new(hf_client);
    let ms_client = Arc::new(ms_client);
    let downloads = futures_util::stream::iter(plans.into_iter().map(|artifact| {
        let cache_root = cache_root.to_path_buf();
        let model_root = model_root.clone();
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
                let target = safe_path(&snapshot_root(&model_root, backend, revision), &path)?;
                link_artifact(&blob, &target)?;
            }
            anyhow::Ok(())
        }
    }))
    .buffer_unordered(concurrency);
    futures_util::pin_mut!(downloads);
    while let Some(result) = downloads.next().await {
        if let Err(error) = result {
            progress.abandon_with_message(format!("{model_id} • download failed"));
            return Err(error);
        }
    }
    progress.finish_with_message(format!("✓ {model_id} • verified backend snapshots"));
    if let Some(manifest) = hf.as_ref() {
        let refs = model_root.join("huggingface").join("refs");
        fs::create_dir_all(&refs)?;
        fs::write(refs.join(huggingface_revision), &manifest.revision)?;
    }
    Ok(DownloadedModel {
        model_root: model_root.clone(),
        huggingface_root: hf_root,
        modelscope_root: ms_root,
    })
}
