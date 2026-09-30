//! Upload files to the Hugging Face Hub.
//!
//! Flow: validate token (`whoami`), create the repository when needed, ask the
//! `preupload` endpoint whether each file is a regular git blob or LFS, upload
//! LFS blobs, then POST an NDJSON commit.

use super::{
    BackendUpload, Diff, LocalFile, LocalHash, UploadBackend, UploadCounts, UploadOptions, plan,
    scope_of, split_repo_id, upload_progress, validate_relative,
};
use crate::repos::RepoKind;
use crate::unified::RemoteEntry;
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::TryStreamExt;
use indicatif::ProgressBar;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE};
use reqwest::{Method, StatusCode};
use serde_json::{Value, json};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Duration;

const OFFICIAL_ENDPOINT: &str = "https://huggingface.co";
const TOKEN_FILE_NAME: &str = "token";
const SAMPLE_BYTES: usize = 512;
const PREUPLOAD_BATCH: usize = 256;
/// Keep inline (base64) content well under the Hub's 1 GiB commit payload cap.
const INLINE_BUDGET_BYTES: u64 = 50 * 1024 * 1024;
const MAX_OPS_PER_COMMIT: usize = 1000;

/// Write endpoint: `HF_ENDPOINT` when set, otherwise the official hub.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or_else(
            || OFFICIAL_ENDPOINT.to_owned(),
            |value| value.trim_end_matches('/').to_owned(),
        )
}

fn home() -> PathBuf {
    std::env::var("HOME").map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
}

/// Locate a Hugging Face write token from the environment or the token file.
pub(crate) fn load_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
    {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_owned());
        }
    }
    let path = std::env::var("HF_TOKEN_PATH")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or_else(
            || {
                std::env::var("HF_HOME")
                    .map_or_else(|_| home().join(".cache").join("huggingface"), PathBuf::from)
                    .join(TOKEN_FILE_NAME)
            },
            PathBuf::from,
        );
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// A file prepared for commit: hashes and the preupload sample.
#[derive(Clone)]
struct Prepared {
    local: LocalFile,
    sha256: String,
    git: String,
    sample: Vec<u8>,
}

/// A commit operation to send to the Hub.
enum HfOp {
    File {
        path: String,
        content: String,
    },
    Lfs {
        path: String,
        oid: String,
        size: u64,
    },
    Delete {
        path: String,
    },
}

/// Fetch the remote file list, treating any failure as an empty repository.
async fn remote_files(
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> BTreeMap<String, RemoteEntry> {
    match crate::unified::fetch_remote_manifest("huggingface", kind, repo_id, revision).await {
        Ok(manifest) => manifest.files,
        Err(error) => {
            tracing::warn!("cannot read remote manifest for {repo_id}@{revision}: {error:#}");
            BTreeMap::new()
        }
    }
}

/// Upload `files` to the Hugging Face repository described by `opts`.
#[allow(clippy::too_many_lines)]
pub(crate) async fn upload(opts: &UploadOptions, files: &[LocalFile]) -> Result<BackendUpload> {
    let token = load_token().context(
        "Hugging Face token not found; set HF_TOKEN or write it to ~/.cache/huggingface/token",
    )?;
    let endpoint = endpoint();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()?;
    let revision = opts.revision.clone().unwrap_or_else(|| "main".to_owned());
    let segment = opts.kind.segment();
    let (namespace, name) = split_repo_id(&opts.repo_id)?;

    // Validate the token before touching the repository.
    let whoami = send(
        &client,
        Method::GET,
        &format!("{endpoint}/api/whoami-v2"),
        &token,
        None,
    )
    .await
    .context("Hugging Face authentication failed")?;
    if whoami.get("name").is_none() {
        bail!("Hugging Face token is not valid (whoami returned no user)");
    }

    // A dry run must not create anything.
    let created = if opts.dry_run {
        false
    } else {
        let created = ensure_repo(&client, &endpoint, &token, opts, namespace, name).await?;
        ensure_revision(
            &client,
            &endpoint,
            &token,
            segment,
            &opts.repo_id,
            &revision,
        )
        .await?;
        created
    };

    let prepared = prepare_all(files)?;
    let hashes: BTreeMap<String, LocalHash> = prepared
        .iter()
        .map(|item| {
            (
                item.local.relative.clone(),
                LocalHash {
                    size: item.local.size,
                    sha256: item.sha256.clone(),
                    git_blob_id: item.git.clone(),
                },
            )
        })
        .collect();

    // Decide what to upload (unless `--force`).
    let (selected, diff) = if opts.force {
        (prepared.clone(), Diff::default())
    } else {
        let remote = remote_files(opts.kind, &opts.repo_id, &revision).await;
        let diff = plan(
            files,
            &hashes,
            &remote,
            scope_of(opts.path_in_repo.as_deref()),
            opts.delete,
        );
        let keep: BTreeSet<&str> = diff
            .to_upload()
            .map(|file| file.relative.as_str())
            .collect();
        let selected = prepared
            .iter()
            .filter(|item| keep.contains(item.local.relative.as_str()))
            .cloned()
            .collect();
        (selected, diff)
    };

    let counts = UploadCounts {
        added: diff.added.len(),
        modified: diff.modified.len(),
        unchanged: diff.unchanged,
        deleted: diff.orphans.len(),
    };
    let total: u64 = selected.iter().map(|item| item.local.size).sum();
    let progress = upload_progress(
        opts.progress,
        UploadBackend::HuggingFace,
        &opts.repo_id,
        total,
    );

    if opts.dry_run {
        progress.finish_and_clear();
        return Ok(BackendUpload {
            backend: UploadBackend::HuggingFace,
            created,
            revision,
            commit: None,
            uploaded: Vec::new(),
            skipped: 0,
            bytes: 0,
            counts,
        });
    }

    // Ask the Hub how each selected file should be uploaded.
    let mut regular: Vec<(LocalFile, String)> = Vec::new();
    let mut lfs: Vec<Prepared> = Vec::new();
    let mut skipped = 0usize;
    for chunk in selected.chunks(PREUPLOAD_BATCH) {
        let modes = preupload(
            &client,
            &endpoint,
            &token,
            segment,
            &opts.repo_id,
            &revision,
            chunk,
        )
        .await?;
        let by_path: HashMap<&str, &Prepared> = chunk
            .iter()
            .map(|item| (item.local.relative.as_str(), item))
            .collect();
        for mode in modes {
            let Some(item) = by_path.get(mode.path.as_str()) else {
                continue;
            };
            if mode.should_ignore {
                skipped += 1;
                continue;
            }
            match mode.mode.as_str() {
                "regular" => {
                    let content = read_inline(&item.local)?;
                    progress.inc(item.local.size);
                    regular.push((item.local.clone(), content));
                }
                "lfs" => lfs.push((*item).clone()),
                "xet" => bail!(
                    "Hugging Face asked to store `{}` with Xet, which modelhub does not support yet; \
                     use `huggingface-cli upload` for this repository",
                    mode.path
                ),
                other => bail!(
                    "unexpected Hugging Face upload mode `{other}` for {}",
                    mode.path
                ),
            }
        }
    }

    // Upload LFS blobs, then build the commit operations.
    let mut operations: Vec<HfOp> = Vec::new();
    let mut bytes = 0u64;
    for item in &lfs {
        upload_lfs(
            &client,
            &endpoint,
            &token,
            opts.kind,
            &opts.repo_id,
            item,
            &progress,
        )
        .await?;
        operations.push(HfOp::Lfs {
            path: item.local.relative.clone(),
            oid: item.sha256.clone(),
            size: item.local.size,
        });
        bytes += item.local.size;
    }
    for (local, content) in regular {
        bytes += local.size;
        operations.push(HfOp::File {
            path: local.relative,
            content,
        });
    }
    for path in &diff.orphans {
        operations.push(HfOp::Delete { path: path.clone() });
    }

    if operations.is_empty() {
        progress.finish_with_message(format!("✓ {} • up to date", opts.repo_id));
        return Ok(BackendUpload {
            backend: UploadBackend::HuggingFace,
            created,
            revision,
            commit: None,
            uploaded: Vec::new(),
            skipped,
            bytes: 0,
            counts,
        });
    }

    let summary = opts
        .commit_message
        .clone()
        .unwrap_or_else(|| format!("Update {} file(s) with modelhub", operations.len()));
    let committer = Committer {
        client: &client,
        endpoint: &endpoint,
        token: &token,
        segment,
        repo_id: &opts.repo_id,
        revision: &revision,
        summary: &summary,
    };
    let (uploaded, commit) = commit_in_batches(&committer, operations).await?;
    progress.finish_with_message(format!(
        "✓ {} • {} file(s) • {}",
        opts.repo_id,
        uploaded.len(),
        revision
    ));

    Ok(BackendUpload {
        backend: UploadBackend::HuggingFace,
        created,
        revision,
        commit,
        uploaded,
        skipped,
        bytes,
        counts,
    })
}

fn read_inline(local: &LocalFile) -> Result<String> {
    let data = std::fs::read(&local.path)
        .with_context(|| format!("cannot read {}", local.path.display()))?;
    Ok(BASE64.encode(data))
}

fn prepare_all(files: &[LocalFile]) -> Result<Vec<Prepared>> {
    files.iter().map(prepare).collect()
}

fn prepare(file: &LocalFile) -> Result<Prepared> {
    validate_relative(&file.relative)?;
    let mut reader = BufReader::new(
        File::open(&file.path).with_context(|| format!("cannot read {}", file.path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", file.size).as_bytes());
    let mut sample = Vec::with_capacity(SAMPLE_BYTES);
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if sample.len() < SAMPLE_BYTES {
            let take = (SAMPLE_BYTES - sample.len()).min(count);
            sample.extend_from_slice(&buffer[..take]);
        }
        hasher.update(&buffer[..count]);
        git.update(&buffer[..count]);
    }
    Ok(Prepared {
        local: file.clone(),
        sha256: format!("{:x}", hasher.finalize()),
        git: format!("{:x}", git.finalize()),
        sample,
    })
}

async fn ensure_repo(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    opts: &UploadOptions,
    namespace: &str,
    name: &str,
) -> Result<bool> {
    let url = format!("{endpoint}/api/{}/{namespace}/{name}", opts.kind.segment());
    let response = client.get(&url).bearer_auth(token).send().await?;
    if response.status().is_success() {
        return Ok(false);
    }
    if response.status() != StatusCode::NOT_FOUND {
        bail!(
            "cannot check Hugging Face repository `{namespace}/{name}`: HTTP {}",
            response.status()
        );
    }
    if !opts.create {
        bail!(
            "Hugging Face repository `{namespace}/{name}` does not exist; pass `create` to create it"
        );
    }
    let repo_type = match opts.kind {
        RepoKind::Model => "model",
        RepoKind::Dataset => "dataset",
    };
    let body = json!({
        "type": repo_type,
        "name": name,
        "organization": namespace,
        "private": opts.private,
    });
    send(
        client,
        Method::POST,
        &format!("{endpoint}/api/repos/create"),
        token,
        Some(body),
    )
    .await
    .with_context(|| format!("failed to create Hugging Face repository `{namespace}/{name}`"))?;
    Ok(true)
}

struct UploadMode {
    path: String,
    mode: String,
    should_ignore: bool,
}

/// Make sure `revision` exists before committing; create it from `main` when
/// missing, matching the `hf upload` CLI. The raw commit API does not create
/// branches on its own.
async fn ensure_revision(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    segment: &str,
    repo_id: &str,
    revision: &str,
) -> Result<()> {
    if revision == "main" {
        return Ok(());
    }
    let encoded = urlencoding::encode(revision);
    let check = client
        .get(format!(
            "{endpoint}/api/{segment}/{repo_id}/revision/{encoded}"
        ))
        .bearer_auth(token)
        .send()
        .await?;
    if check.status().is_success() {
        return Ok(());
    }
    if check.status() != StatusCode::NOT_FOUND {
        let status = check.status();
        let text = check.text().await.unwrap_or_default();
        bail!("cannot resolve Hugging Face revision `{revision}`: HTTP {status}: {text}");
    }
    let create = client
        .post(format!(
            "{endpoint}/api/{segment}/{repo_id}/branch/{encoded}"
        ))
        .bearer_auth(token)
        .json(&json!({"startingPoint": "main"}))
        .send()
        .await?;
    if create.status().is_success() || create.status() == StatusCode::CONFLICT {
        Ok(())
    } else {
        let status = create.status();
        let text = create.text().await.unwrap_or_default();
        bail!("failed to create Hugging Face branch `{revision}`: HTTP {status}: {text}")
    }
}

async fn preupload(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    segment: &str,
    repo_id: &str,
    revision: &str,
    chunk: &[Prepared],
) -> Result<Vec<UploadMode>> {
    let body = json!({
        "files": chunk.iter().map(|item| json!({
            "path": item.local.relative,
            "sample": BASE64.encode(&item.sample),
            "size": item.local.size,
        })).collect::<Vec<_>>(),
    });
    let url = format!(
        "{endpoint}/api/{segment}/{repo_id}/preupload/{}",
        urlencoding::encode(revision)
    );
    let value = send(client, Method::POST, &url, token, Some(body)).await?;
    let files = value
        .get("files")
        .and_then(Value::as_array)
        .context("Hugging Face preupload returned no files")?;
    Ok(files
        .iter()
        .map(|file| UploadMode {
            path: file
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            mode: file
                .get("uploadMode")
                .and_then(Value::as_str)
                .unwrap_or("regular")
                .to_owned(),
            should_ignore: file
                .get("shouldIgnore")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
        .collect())
}

fn lfs_prefix(kind: RepoKind) -> &'static str {
    match kind {
        RepoKind::Model => "",
        RepoKind::Dataset => "datasets/",
    }
}

async fn upload_lfs(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    kind: RepoKind,
    repo_id: &str,
    item: &Prepared,
    progress: &ProgressBar,
) -> Result<()> {
    let batch_url = format!(
        "{endpoint}/{}{repo_id}.git/info/lfs/objects/batch",
        lfs_prefix(kind)
    );
    let body = json!({
        "operation": "upload",
        "transfers": ["basic", "multipart"],
        "objects": [{"oid": item.sha256, "size": item.local.size}],
        "hash_algo": "sha256",
    });
    let response = client
        .post(&batch_url)
        .bearer_auth(token)
        .header(CONTENT_TYPE, "application/vnd.git-lfs+json")
        .header("Accept", "application/vnd.git-lfs+json")
        .json(&body)
        .send()
        .await?;
    if !response.status().is_success() {
        bail!(
            "Hugging Face LFS batch failed for {}: HTTP {}",
            item.local.relative,
            response.status()
        );
    }
    let value: Value = response.json().await?;
    let object = value
        .get("objects")
        .and_then(Value::as_array)
        .and_then(|objects| objects.first())
        .context("Hugging Face LFS batch returned no object")?;
    let Some(upload) = object.pointer("/actions/upload") else {
        // No actions means the content already exists on the Hub.
        progress.inc(item.local.size);
        return Ok(());
    };
    let href = upload
        .get("href")
        .and_then(Value::as_str)
        .context("Hugging Face LFS upload action has no href")?;
    let header = upload.get("header").cloned().unwrap_or(Value::Null);
    match header
        .get("chunk_size")
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<u64>().ok())
    {
        Some(chunk_size) => {
            upload_lfs_multipart(
                client,
                href,
                &header,
                &item.local,
                chunk_size,
                &item.sha256,
                progress,
            )
            .await
        }
        None => upload_lfs_single(client, href, &item.local, progress).await,
    }
    .with_context(|| format!("failed to upload LFS file {}", item.local.relative))?;

    // Optional verify step.
    if let Some(verify) = object
        .pointer("/actions/verify/href")
        .and_then(Value::as_str)
    {
        let _ = client
            .post(verify)
            .bearer_auth(token)
            .json(&json!({"oid": item.sha256, "size": item.local.size}))
            .send()
            .await;
    }
    Ok(())
}

async fn upload_lfs_single(
    client: &reqwest::Client,
    href: &str,
    local: &LocalFile,
    progress: &ProgressBar,
) -> Result<()> {
    let file = tokio::fs::File::open(&local.path).await?;
    let progress = progress.clone();
    let stream = tokio_util::io::ReaderStream::new(file).map_ok(move |chunk| {
        progress.inc(chunk.len() as u64);
        chunk
    });
    let response = client
        .put(href)
        .header(CONTENT_LENGTH, local.size)
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await?;
    if !response.status().is_success() {
        bail!("LFS PUT returned HTTP {}", response.status());
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
async fn upload_lfs_multipart(
    client: &reqwest::Client,
    href: &str,
    header: &Value,
    local: &LocalFile,
    chunk_size: u64,
    oid: &str,
    progress: &ProgressBar,
) -> Result<()> {
    let mut parts: Vec<(u64, String)> = header
        .as_object()
        .map(|map| {
            map.iter()
                .filter_map(|(key, value)| {
                    let number = key.parse::<u64>().ok()?;
                    let url = value.as_str()?.to_owned();
                    Some((number, url))
                })
                .collect()
        })
        .unwrap_or_default();
    parts.sort_by_key(|(number, _)| *number);
    if parts.is_empty() {
        bail!("LFS multipart response had no part URLs");
    }
    let mut file = File::open(&local.path)?;
    let mut completion = Vec::with_capacity(parts.len());
    for (index, (_, url)) in parts.iter().enumerate() {
        let offset = chunk_size * index as u64;
        let length = chunk_size.min(local.size.saturating_sub(offset));
        file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer)?;
        let response = client.put(url).body(buffer).send().await?;
        if !response.status().is_success() {
            bail!("LFS part {} returned HTTP {}", index + 1, response.status());
        }
        progress.inc(length);
        let etag = response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        completion.push(json!({"partNumber": index + 1, "etag": etag}));
    }
    let response = client
        .post(href)
        .json(&json!({"oid": oid, "parts": completion}))
        .send()
        .await?;
    if !response.status().is_success() {
        bail!("LFS completion returned HTTP {}", response.status());
    }
    Ok(())
}

struct Committer<'a> {
    client: &'a reqwest::Client,
    endpoint: &'a str,
    token: &'a str,
    segment: &'a str,
    repo_id: &'a str,
    revision: &'a str,
    summary: &'a str,
}

/// Commit operations, splitting into several commits when the inline payload
/// or the operation count grows too large. Returns uploaded paths and the last
/// commit id.
async fn commit_in_batches(
    committer: &Committer<'_>,
    ops: Vec<HfOp>,
) -> Result<(Vec<String>, Option<String>)> {
    let mut uploaded = Vec::new();
    let mut batch: Vec<HfOp> = Vec::new();
    let mut inline_bytes = 0u64;
    let mut commit = None;
    for op in ops {
        let size = match &op {
            HfOp::File { content, .. } => content.len() as u64,
            HfOp::Lfs { .. } | HfOp::Delete { .. } => 0,
        };
        if !batch.is_empty()
            && (inline_bytes + size > INLINE_BUDGET_BYTES || batch.len() >= MAX_OPS_PER_COMMIT)
        {
            commit = commit_one(committer, &batch).await?.or(commit);
            batch.clear();
            inline_bytes = 0;
        }
        inline_bytes += size;
        if let HfOp::File { path, .. } | HfOp::Lfs { path, .. } = &op {
            uploaded.push(path.clone());
        }
        batch.push(op);
    }
    if !batch.is_empty() {
        commit = commit_one(committer, &batch).await?.or(commit);
    }
    Ok((uploaded, commit))
}

async fn commit_one(committer: &Committer<'_>, ops: &[HfOp]) -> Result<Option<String>> {
    let payload = commit_ndjson(ops, committer.summary, None);
    let url = format!(
        "{}/api/{}/{}/commit/{}",
        committer.endpoint,
        committer.segment,
        committer.repo_id,
        urlencoding::encode(committer.revision)
    );
    let response = committer
        .client
        .post(&url)
        .bearer_auth(committer.token)
        .header(CONTENT_TYPE, "application/x-ndjson")
        .body(payload)
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("Hugging Face commit failed: HTTP {status}: {text}");
    }
    let value: Value = response.json().await.unwrap_or(Value::Null);
    Ok(value
        .get("commitOid")
        .and_then(Value::as_str)
        .map(str::to_owned))
}

/// Build the NDJSON commit body the Hub expects.
fn commit_ndjson(ops: &[HfOp], summary: &str, description: Option<&str>) -> String {
    let mut header = json!({"key": "header", "value": {"summary": summary}});
    if let Some(description) = description {
        header["value"]["description"] = json!(description);
    }
    let mut lines = vec![header.to_string()];
    for op in ops {
        let line = match op {
            HfOp::File { path, content } => json!({
                "key": "file",
                "value": {"content": content, "path": path, "encoding": "base64"},
            }),
            HfOp::Lfs { path, oid, size } => json!({
                "key": "lfsFile",
                "value": {"path": path, "algo": "sha256", "oid": oid, "size": size},
            }),
            HfOp::Delete { path } => json!({
                "key": "deletedFile",
                "value": {"path": path},
            }),
        };
        lines.push(line.to_string());
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

async fn send(
    client: &reqwest::Client,
    method: Method,
    url: &str,
    token: &str,
    body: Option<Value>,
) -> Result<Value> {
    let mut request = client.request(method, url).bearer_auth(token);
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("HTTP {status} from {url}: {text}");
    }
    Ok(response.json().await.unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::{HfOp, commit_ndjson};

    #[test]
    fn builds_ndjson_commit() {
        let ops = vec![
            HfOp::File {
                path: "config.json".to_owned(),
                content: "e30=".to_owned(),
            },
            HfOp::Lfs {
                path: "model.safetensors".to_owned(),
                oid: "abc".to_owned(),
                size: 42,
            },
        ];
        let payload = commit_ndjson(&ops, "first commit", None);
        let lines: Vec<_> = payload.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[0],
            r#"{"key":"header","value":{"summary":"first commit"}}"#
        );
        assert_eq!(
            lines[1],
            r#"{"key":"file","value":{"content":"e30=","encoding":"base64","path":"config.json"}}"#
        );
        assert_eq!(
            lines[2],
            r#"{"key":"lfsFile","value":{"algo":"sha256","oid":"abc","path":"model.safetensors","size":42}}"#
        );
        assert!(payload.ends_with('\n'));
    }

    #[test]
    fn description_is_included() {
        let payload = commit_ndjson(&[], "msg", Some("detail"));
        assert!(payload.contains("\"description\":\"detail\""));
    }

    #[test]
    fn emits_delete_operations() {
        let payload = commit_ndjson(
            &[HfOp::Delete {
                path: "old.bin".to_owned(),
            }],
            "sync",
            None,
        );
        assert!(payload.contains(r#"{"key":"deletedFile","value":{"path":"old.bin"}}"#));
    }
}
