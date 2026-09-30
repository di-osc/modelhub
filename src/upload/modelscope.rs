//! Upload files to `ModelScope`.
//!
//! Flow: create the repository when needed, upload large files through the LFS
//! batch/blob endpoints, then POST a single (or split) commit describing every
//! file as an inline base64 blob or an LFS pointer.

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
use reqwest::header::CONTENT_LENGTH;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_ENDPOINT: &str = "https://modelscope.cn";
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36";
/// Files above this size go through LFS regardless of suffix.
const LFS_FORCE_BYTES: u64 = 64 * 1024;
/// Server cap on actions in one commit request.
const MAX_ACTIONS_PER_COMMIT: usize = 2000;
/// Keep inline base64 content under a conservative request budget.
const INLINE_BUDGET_BYTES: u64 = 50 * 1024 * 1024;
/// Files whose names the Hub parses server-side and must stay inline.
const INLINE_METADATA_NAMES: &[&str] = &[
    "README.MD",
    ".GITATTRIBUTES",
    ".GITIGNORE",
    "CONFIGURATION.JSON",
    "CONFIGURATION.YAML",
    "CONFIGURATION.YML",
    "DATASET_INFOS.JSON",
    "CONFIG.JSON",
    ".MSC",
    ".MDL",
];
/// Extensions stored through LFS by `ModelScope` (model + dataset lists).
const LFS_SUFFIXES: &[&str] = &[
    ".7z",
    ".aac",
    ".arrow",
    ".audio",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".flac",
    ".ftz",
    ".gif",
    ".gz",
    ".h5",
    ".jack",
    ".jpeg",
    ".joblib",
    ".jpg",
    ".jsonl",
    ".mlmodel",
    ".model",
    ".msgpack",
    ".npy",
    ".npz",
    ".onnx",
    ".ot",
    ".parquet",
    ".pb",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".tar",
    ".tflite",
    ".tgz",
    ".wasm",
    ".xz",
    ".zip",
    ".zst",
];

fn endpoint() -> String {
    std::env::var("MODELSCOPE_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or_else(
            || DEFAULT_ENDPOINT.to_owned(),
            |value| value.trim_end_matches('/').to_owned(),
        )
}

fn home() -> PathBuf {
    std::env::var("HOME").map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
}

/// Locate a `ModelScope` token.
///
/// Order: `MODELSCOPE_API_TOKEN`/`MODELSCOPE_TOKEN` environment variables, the
/// pickled cookie jar written by `modelscope login`, then the legacy JSON
/// cookies file. `~/.modelscope/credentials/session` is an anonymous SDK
/// install identifier and is deliberately never treated as a token.
pub(crate) fn load_token() -> Option<String> {
    for name in ["MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"] {
        if let Ok(token) = std::env::var(name) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_owned());
            }
        }
    }
    let credentials = home().join(".modelscope").join("credentials");
    if let Ok(data) = std::fs::read(credentials.join("cookies"))
        && let Some(token) = token_from_pickle(&data)
    {
        return Some(token);
    }
    let cookies = home().join(".modelscope").join("config").join("cookies");
    let data = std::fs::read_to_string(cookies).ok()?;
    let value: Value = serde_json::from_str(&data).ok()?;
    value
        .get("m_session_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Extract the `m_session_id` cookie from a pickled `CookieJar`.
///
/// The jar pickles each cookie as a mapping. Depending on how the pickle was
/// built the `value` key may be memoised, so instead of relying on that we scan
/// every pickle string and pick the token: modern `ModelScope` tokens start
/// with `ms-`, and the `m_session_id` value is preferred when several
/// candidates exist.
fn token_from_pickle(data: &[u8]) -> Option<String> {
    let strings = pickle_strings(data);
    let anchor = strings.iter().position(|value| value == "m_session_id");
    let is_ms = |value: &str| value.starts_with("ms-") && value.len() >= 8;
    let is_token_like = |value: &str| {
        (16..=128).contains(&value.len())
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            && value.chars().any(|c| c.is_ascii_alphanumeric())
    };
    let after = |predicate: &dyn Fn(&str) -> bool| {
        anchor.and_then(|anchor| {
            strings
                .iter()
                .skip(anchor + 1)
                .take(4)
                .find(|value| predicate(value))
                .cloned()
        })
    };
    after(&is_ms)
        .or_else(|| strings.iter().find(|value| is_ms(value)).cloned())
        .or_else(|| after(&is_token_like))
}

/// Collect every text string in a pickle byte stream.
fn pickle_strings(data: &[u8]) -> Vec<String> {
    let mut strings = Vec::new();
    let mut position = 0;
    while position < data.len() {
        if let Some((value, next)) = pickle_string_at(data, position) {
            strings.push(value);
            position = next;
        } else {
            position += 1;
        }
    }
    strings
}

/// Read a pickle text string at `position`, returning it and the offset after.
fn pickle_string_at(data: &[u8], position: usize) -> Option<(String, usize)> {
    let opcode = *data.get(position)?;
    let (length, header) = match opcode {
        // SHORT_BINUNICODE / SHORT_BINSTRING: 1-byte length.
        0x8c | 0x55 => (*data.get(position + 1)? as usize, 2),
        // BINUNICODE / BINSTRING: 4-byte little-endian length.
        0x58 | 0x54 => {
            let bytes: [u8; 4] = data.get(position + 1..position + 5)?.try_into().ok()?;
            (u32::from_le_bytes(bytes) as usize, 5)
        }
        // BINUNICODE8: 8-byte little-endian length.
        0x8d => {
            let bytes: [u8; 8] = data.get(position + 1..position + 9)?.try_into().ok()?;
            (usize::try_from(u64::from_le_bytes(bytes)).ok()?, 9)
        }
        _ => return None,
    };
    let start = position + header;
    let text = data.get(start..start + length)?;
    Some((String::from_utf8_lossy(text).into_owned(), start + length))
}

/// A `ModelScope` commit action that removes a remote file.
fn delete_action(path: &str) -> Value {
    json!({
        "action": "delete",
        "path": path,
        "type": "normal",
        "size": 0,
        "sha256": "",
        "content": "",
        "encoding": "",
    })
}

/// Whether a file is stored through LFS on `ModelScope`.
fn is_lfs(name: &str, size: u64) -> bool {
    let upper = name.to_ascii_uppercase();
    if INLINE_METADATA_NAMES.contains(&upper.as_str()) {
        return false;
    }
    if size > LFS_FORCE_BYTES {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    LFS_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix))
}

struct Prepared {
    local: LocalFile,
    sha256: String,
}

/// Upload `files` to the `ModelScope` repository described by `opts`.
#[allow(clippy::too_many_lines)]
pub(crate) async fn upload(opts: &UploadOptions, files: &[LocalFile]) -> Result<BackendUpload> {
    let token = load_token().context(
        "ModelScope token not found; set MODELSCOPE_API_TOKEN or run `modelscope login`",
    )?;
    let endpoint = endpoint();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()?;
    let revision = opts.revision.clone().unwrap_or_else(|| "master".to_owned());
    let segment = opts.kind.segment();
    let (namespace, name) = split_repo_id(&opts.repo_id)?;

    let created = if opts.dry_run {
        false
    } else {
        ensure_repo(&client, &endpoint, &token, opts, namespace, name).await?
    };

    // Hash every file so unchanged ones can be skipped.
    let mut hashes: BTreeMap<String, LocalHash> = BTreeMap::new();
    for file in files {
        validate_relative(&file.relative)?;
        hashes.insert(
            file.relative.clone(),
            LocalHash {
                size: file.size,
                sha256: sha256_file(&file.path)?,
                git_blob_id: String::new(),
            },
        );
    }

    let (selected, diff) = if opts.force {
        (files.to_vec(), Diff::default())
    } else {
        let remote = remote_files(opts.kind, &opts.repo_id, &revision).await;
        let diff = plan(
            files,
            &hashes,
            &remote,
            scope_of(opts.path_in_repo.as_deref()),
            opts.delete,
        );
        let selected = diff.to_upload().cloned().collect();
        (selected, diff)
    };

    let counts = UploadCounts {
        added: diff.added.len(),
        modified: diff.modified.len(),
        unchanged: diff.unchanged,
        deleted: diff.orphans.len(),
    };
    let total: u64 = selected.iter().map(|file| file.size).sum();
    let progress = upload_progress(
        opts.progress,
        UploadBackend::ModelScope,
        &opts.repo_id,
        total,
    );

    if opts.dry_run {
        progress.finish_and_clear();
        return Ok(BackendUpload {
            backend: UploadBackend::ModelScope,
            created,
            revision,
            commit: None,
            uploaded: Vec::new(),
            skipped: 0,
            bytes: 0,
            counts,
        });
    }

    // Split selected files into LFS and inline.
    let mut lfs: Vec<Prepared> = Vec::new();
    let mut normal: Vec<LocalFile> = Vec::new();
    for file in &selected {
        if is_lfs(&file.relative, file.size) {
            let sha256 = hashes
                .get(&file.relative)
                .map_or_else(|| sha256_file(&file.path), |hash| Ok(hash.sha256.clone()))?;
            lfs.push(Prepared {
                local: file.clone(),
                sha256,
            });
        } else {
            normal.push(file.clone());
        }
    }

    // Upload LFS blobs.
    for item in &lfs {
        upload_lfs(
            &client,
            &endpoint,
            &token,
            segment,
            &opts.repo_id,
            item,
            &progress,
        )
        .await?;
    }

    // Build actions.
    let mut actions = Vec::with_capacity(selected.len() + diff.orphans.len());
    let mut bytes = 0u64;
    for item in &lfs {
        bytes += item.local.size;
        actions.push(json!({
            "action": "create",
            "path": item.local.relative,
            "type": "lfs",
            "size": item.local.size,
            "sha256": item.sha256,
            "content": "",
            "encoding": "",
        }));
    }
    for file in &normal {
        let content = BASE64.encode(
            std::fs::read(&file.path)
                .with_context(|| format!("cannot read {}", file.path.display()))?,
        );
        bytes += file.size;
        progress.inc(file.size);
        actions.push(json!({
            "action": "create",
            "path": file.relative,
            "type": "normal",
            "size": file.size,
            "sha256": "",
            "content": content,
            "encoding": "base64",
        }));
    }
    for path in &diff.orphans {
        actions.push(delete_action(path));
    }

    if actions.is_empty() {
        progress.finish_with_message(format!("✓ {} • up to date", opts.repo_id));
        return Ok(BackendUpload {
            backend: UploadBackend::ModelScope,
            created,
            revision,
            commit: None,
            uploaded: Vec::new(),
            skipped: 0,
            bytes: 0,
            counts,
        });
    }

    let message = opts
        .commit_message
        .clone()
        .unwrap_or_else(|| format!("Update {} file(s) with modelhub", actions.len()));
    let commit = commit_all(
        &client,
        &endpoint,
        &token,
        segment,
        &opts.repo_id,
        &revision,
        &message,
        actions,
    )
    .await?;

    progress.finish_with_message(format!(
        "✓ {} • {} file(s) • {revision}",
        opts.repo_id,
        selected.len()
    ));
    Ok(BackendUpload {
        backend: UploadBackend::ModelScope,
        created,
        revision,
        commit,
        uploaded: selected.iter().map(|file| file.relative.clone()).collect(),
        skipped: 0,
        bytes,
        counts,
    })
}

/// Fetch the remote file list, treating any failure as an empty repository.
async fn remote_files(
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> BTreeMap<String, RemoteEntry> {
    match crate::unified::fetch_remote_manifest("modelscope", kind, repo_id, revision).await {
        Ok(manifest) => manifest.files,
        Err(error) => {
            tracing::warn!("cannot read remote manifest for {repo_id}@{revision}: {error:#}");
            BTreeMap::new()
        }
    }
}

fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("cannot read {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Auth headers for every `ModelScope` request.
fn auth(request: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
    request
        .header("Cookie", format!("m_session_id={token}"))
        .header("Authorization", format!("Bearer {token}"))
        .header("User-Agent", USER_AGENT)
}

async fn ensure_repo(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    opts: &UploadOptions,
    namespace: &str,
    name: &str,
) -> Result<bool> {
    let segment = opts.kind.segment();
    let check = auth(
        client.get(format!("{endpoint}/api/v1/{segment}/{namespace}/{name}")),
        token,
    )
    .send()
    .await?;
    if check.status().is_success() {
        return Ok(false);
    }
    if check.status() != reqwest::StatusCode::NOT_FOUND {
        let status = check.status();
        let text = check.text().await.unwrap_or_default();
        bail!("cannot check ModelScope repository `{namespace}/{name}`: HTTP {status}: {text}");
    }
    if !opts.create {
        bail!(
            "ModelScope repository `{namespace}/{name}` does not exist; pass `create` to create it"
        );
    }
    // PRIVATE = 1, PUBLIC = 5.
    let visibility = if opts.private { 1 } else { 5 };
    let response = match opts.kind {
        RepoKind::Model => {
            let body = json!({
                "Path": namespace,
                "Name": name,
                "Visibility": visibility,
                "License": "Apache-2.0",
            });
            auth(client.post(format!("{endpoint}/api/v1/models")), token)
                .json(&body)
                .send()
                .await?
        }
        RepoKind::Dataset => {
            let form = reqwest::multipart::Form::new()
                .text("Owner", namespace.to_owned())
                .text("Name", name.to_owned())
                .text("Visibility", visibility.to_string())
                .text("License", "Apache-2.0".to_owned());
            auth(client.post(format!("{endpoint}/api/v1/datasets")), token)
                .multipart(form)
                .send()
                .await?
        }
    };
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("failed to create ModelScope repository `{namespace}/{name}`: HTTP {status}: {text}");
    }
    Ok(true)
}

async fn upload_lfs(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    segment: &str,
    repo_id: &str,
    item: &Prepared,
    progress: &ProgressBar,
) -> Result<()> {
    let batch_url = format!("{endpoint}/api/v1/repos/{segment}/{repo_id}/info/lfs/objects/batch");
    let body = json!({
        "operation": "upload",
        "objects": [{"oid": item.sha256, "size": item.local.size}],
    });
    let response = auth(client.post(&batch_url), token)
        .json(&body)
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!(
            "ModelScope LFS batch failed for {}: HTTP {status}: {text}",
            item.local.relative
        );
    }
    let value: Value = response.json().await?;
    let data = value.get("Data").unwrap_or(&value);
    let object = data
        .get("objects")
        .and_then(Value::as_array)
        .and_then(|objects| objects.first());
    let Some(href) = object
        .and_then(|object| object.pointer("/actions/upload/href"))
        .and_then(Value::as_str)
    else {
        // No upload action means the blob already exists.
        progress.inc(item.local.size);
        return Ok(());
    };
    let file = tokio::fs::File::open(&item.local.path).await?;
    let progress = progress.clone();
    let stream = tokio_util::io::ReaderStream::new(file).map_ok(move |chunk| {
        progress.inc(chunk.len() as u64);
        chunk
    });
    let response = auth(client.put(href), token)
        .header(CONTENT_LENGTH, item.local.size)
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await?;
    if !response.status().is_success() {
        bail!(
            "ModelScope LFS upload failed for {}: HTTP {}",
            item.local.relative,
            response.status()
        );
    }
    Ok(())
}

/// Commit all actions, splitting when the server limits are approached.
#[allow(clippy::too_many_arguments)]
async fn commit_all(
    client: &reqwest::Client,
    endpoint: &str,
    token: &str,
    segment: &str,
    repo_id: &str,
    revision: &str,
    message: &str,
    actions: Vec<Value>,
) -> Result<Option<String>> {
    let url = format!(
        "{endpoint}/api/v1/repos/{segment}/{repo_id}/commit/{}",
        urlencoding::encode(revision)
    );
    let mut batch: Vec<Value> = Vec::new();
    let mut inline_bytes = 0u64;
    let mut commit = None;
    for action in actions {
        let size = action
            .get("content")
            .and_then(Value::as_str)
            .map_or(0, |content| content.len() as u64);
        if !batch.is_empty()
            && (inline_bytes + size > INLINE_BUDGET_BYTES || batch.len() >= MAX_ACTIONS_PER_COMMIT)
        {
            commit = commit_one(client, &url, token, message, &batch)
                .await?
                .or(commit);
            batch.clear();
            inline_bytes = 0;
        }
        inline_bytes += size;
        batch.push(action);
    }
    if !batch.is_empty() {
        commit = commit_one(client, &url, token, message, &batch)
            .await?
            .or(commit);
    }
    Ok(commit)
}

async fn commit_one(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    message: &str,
    actions: &[Value],
) -> Result<Option<String>> {
    let body = json!({"commit_message": message, "actions": actions});
    let response = auth(client.post(url), token).json(&body).send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("ModelScope commit failed: HTTP {status}: {text}");
    }
    let value: Value = response.json().await.unwrap_or(Value::Null);
    if value.get("Success").and_then(Value::as_bool) == Some(false) {
        bail!(
            "ModelScope commit rejected: {}",
            value
                .get("Message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        );
    }
    let data = value.get("Data").cloned().unwrap_or(Value::Null);
    Ok(data
        .get("CommitId")
        .or_else(|| data.get("commitId"))
        .and_then(Value::as_str)
        .map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::{delete_action, is_lfs, token_from_pickle};

    #[test]
    fn builds_delete_action() {
        let action = delete_action("old/file.bin");
        assert_eq!(action["action"], "delete");
        assert_eq!(action["path"], "old/file.bin");
        assert_eq!(action["type"], "normal");
        assert_eq!(action["size"], 0);
    }

    #[test]
    fn lfs_decision_follows_size_and_suffix() {
        // Inline metadata stays normal even when large.
        assert!(!is_lfs("README.md", 10 * 1024 * 1024));
        assert!(!is_lfs("config.json", 10 * 1024 * 1024));
        // Large files always go to LFS.
        assert!(is_lfs("weights.bin", 2 * 1024 * 1024));
        // Small files use the suffix list.
        assert!(is_lfs("model.safetensors", 10));
        assert!(!is_lfs("train.py", 10));
    }

    #[test]
    fn extracts_token_from_pickled_cookie_jar() {
        // Mirrors the bytes `modelscope_hub` writes via `pickle.dumps(jar)`.
        let mut data = Vec::new();
        data.extend_from_slice(
            b"\x80\x04\x95\xd2\x02\x00\x00\x00\x00\x00\x00]\x94}\x94(\x8c\x0cm_session_id\x94",
        );
        data.extend_from_slice(b"\x8c\x06Cookie\x94}\x94(\x8c\x07version\x94K\x00\x8c\x04name\x94");
        data.push(0x8c);
        data.push(0x05);
        data.extend_from_slice(b"value");
        data.push(0x8c);
        data.push(0x0b);
        data.extend_from_slice(b"ms-TOKEN123");
        assert_eq!(token_from_pickle(&data).as_deref(), Some("ms-TOKEN123"));
    }

    #[test]
    fn extracts_ms_token_when_value_key_is_memoised() {
        // Matches the real `modelscope login` jar: a 60-hex cookie value, a
        // memoised `value` key, then the `m_session_id` token.
        let mut data = Vec::new();
        data.push(0x8c);
        data.push(7);
        data.extend_from_slice(b"value");
        data.push(0x8c);
        data.push(60);
        data.extend_from_slice(b"2f5c8ddc17907354778136201e23ba3e294bca95bd20920606773826f94b");
        data.push(0x8c);
        data.push(12);
        data.extend_from_slice(b"m_session_id");
        data.extend_from_slice(b"\x94h$)\x81\x94}\x94(h'K\x00h(hGh)");
        let token = b"ms-a6f3d523-8c0c-42f5-afe9-e2fb45a6b1bb";
        data.push(0x8c);
        data.push(u8::try_from(token.len()).unwrap());
        data.extend_from_slice(token);
        assert_eq!(
            token_from_pickle(&data).as_deref(),
            Some("ms-a6f3d523-8c0c-42f5-afe9-e2fb45a6b1bb")
        );
    }

    #[test]
    fn missing_token_file_is_not_a_token() {
        assert!(token_from_pickle(b"no pickle here").is_none());
    }
}
