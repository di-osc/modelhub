# 单文件下载快速路径实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `modelhub::download` 在 `file` 有值时只请求这一个文件（缓存命中零网络、绝不列举仓库），并让 `kind`/`backend` 可选地消除自动探测。

**Architecture:** 复用现有 content-addressed 下载管线（staging → `blobs/sha256/<hash>` → hard-link 到 `{cache}/{kind}/{repo--id}/{backend}/snapshots/{revision}/{file}`）。把 `materialize` 拆出 `stream_to_blob`，让单文件路径能把"已经拿到的响应"直接落盘。类型/后端提示收窄现有的清单探测，而不是新增第二个下载入口。

**Tech Stack:** Rust 2024、tokio、reqwest、futures-util、anyhow、clap；集成测试用本地手写 mock HTTP server（`std::net`）+ `temp_env`（edition 2024 下 `std::env::set_var` 是 unsafe，而本 crate `unsafe_code = "forbid"`）。

---

## 文件结构

| 文件 | 责任 |
|---|---|
| `src/unified.rs` | `Backend` 公开、单文件 URL 构造、路径校验、`stream_to_blob`、`download_single_file`、清单探测收窄 |
| `src/ops.rs` | `DownloadOptions` 新字段、`download()` 分派 |
| `src/lib.rs` | 导出 `Backend` |
| `src/main.rs` | CLI `--repo-type` / `--backend` |
| `tests/download.rs` | mock hub + 单文件/整仓收窄集成测试 |
| `Cargo.toml` | 增加 `temp_env` dev-dependency |
| `README.md` | 用法说明 |

---

### Task 1: 公开 `Backend`、单文件 URL 与 `MODELSCOPE_ENDPOINT`

**Files:**
- Modify: `src/unified.rs:25-38`（`Backend` 定义）、`src/unified.rs:302-307`（常量）、`src/unified.rs:374-442`（URL 拼装）、文件末尾新增 `#[cfg(test)] mod tests`
- Modify: `src/lib.rs:37-38`

- [ ] **Step 1: 写失败的单元测试**

在 `src/unified.rs` 末尾追加：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huggingface_file_url_encodes_each_segment() {
        let url = huggingface_file_url(
            "https://hf.example",
            RepoKind::Dataset,
            "org/name",
            "main",
            "data/a b.mp3",
        );
        assert_eq!(
            url,
            "https://hf.example/datasets/org/name/resolve/main/data/a%20b.mp3"
        );
    }

    #[test]
    fn modelscope_file_url_targets_one_file() {
        let url = modelscope_file_url(RepoKind::Dataset, "org/name", "master", "data/a.mp3");
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
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --lib unified::tests`
Expected: 编译失败，`cannot find function huggingface_file_url`（以及 `modelscope_file_url`、`default_revision`）

- [ ] **Step 3: 实现 URL 与枚举改动**

把 `src/unified.rs` 里的：

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    HuggingFace,
    ModelScope,
}

impl Backend {
    const fn segment(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }
}
```

替换为：

```rust
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
```

在 `const DATASET_PAGE_SIZE: usize = 200;` 下面加：

```rust
const MS_OFFICIAL: &str = "https://modelscope.cn";
```

在 `fn hf_endpoints()` 之后加：

```rust
/// Base URL for `ModelScope` requests; `MODELSCOPE_ENDPOINT` overrides it.
fn ms_base_url() -> String {
    std::env::var("MODELSCOPE_ENDPOINT").map_or_else(
        |_| MS_OFFICIAL.to_owned(),
        |value| value.trim_end_matches('/').to_owned(),
    )
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
        "{endpoint}/{}/{}/resolve/{}/{}",
        kind.segment(),
        encode_path(repo_id),
        encode_path(revision),
        encode_path(file)
    )
}

/// Single-file download URL on `ModelScope`.
fn modelscope_file_url(kind: RepoKind, repo_id: &str, revision: &str, file: &str) -> String {
    format!(
        "{}/api/v1/{}/{}/repo?Revision={}&FilePath={}",
        ms_base_url(),
        kind.segment(),
        repo_id,
        urlencoding::encode(revision),
        urlencoding::encode(file)
    )
}
```

把 `modelscope_manifest` 里的 url 字段：

```rust
                url: format!(
                    "https://modelscope.cn/api/v1/{}/{}/repo?Revision={}&FilePath={}",
                    kind.segment(),
                    repo_id,
                    urlencoding::encode(revision),
                    urlencoding::encode(&path)
                ),
```

替换为：

```rust
                url: modelscope_file_url(kind, repo_id, revision, &path),
```

把 `modelscope_list_files` 的两个硬编码域名替换为 `ms_base_url()`：

```rust
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
```

把 `src/lib.rs` 的：

```rust
/// Result of [`download`].
pub use unified::DownloadedRepo;
```

替换为：

```rust
/// Result of [`download`].
pub use unified::DownloadedRepo;

/// Hub that hosts a repository.
pub use unified::Backend;
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cargo test --lib unified::tests`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/unified.rs src/lib.rs
git commit -m "Expose Backend and add single-file URL builders with ModelScope endpoint override"
```

---

### Task 2: 路径与 revision 校验

**Files:**
- Modify: `src/unified.rs`（`safe_path` 附近新增函数；`#[cfg(test)] mod tests` 追加）
- Test: `src/unified.rs` 内联单元测试

- [ ] **Step 1: 写失败的单元测试**

在 `src/unified.rs` 的 `mod tests` 里追加：

```rust
    #[test]
    fn validate_relative_rejects_traversal_and_absolute_paths() {
        assert!(validate_relative("file path", "data/a.mp3").is_ok());
        assert!(validate_relative("revision", "refs/pr/1").is_ok());
        assert!(validate_relative("file path", "../evil").is_err());
        assert!(validate_relative("file path", "a/../../b").is_err());
        assert!(validate_relative("file path", "/etc/passwd").is_err());
        assert!(validate_relative("revision", "..").is_err());
        assert!(validate_relative("revision", "/main").is_err());
        assert!(validate_relative("file path", "").is_err());
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --lib validate_relative`
Expected: 编译失败，`cannot find function validate_relative`

- [ ] **Step 3: 实现校验**

在 `fn safe_path` 之前插入：

```rust
/// Reject empty, absolute, or `..`-containing repository paths and revisions.
///
/// Callers run this before any network request so a bad `file` or `revision`
/// never reaches a hub.
fn validate_relative(label: &str, value: &str) -> anyhow::Result<()> {
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cargo test --lib validate_relative`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/unified.rs
git commit -m "Add path and revision validation for repository-relative values"
```

---

### Task 3: 拆出 `stream_to_blob` 与进度条 helper（行为不变的重构）

**Files:**
- Modify: `src/unified.rs:508-611`（`materialize`）、`src/unified.rs:844-854`（`download_loaded` 的进度条）

- [ ] **Step 1: 记录重构前的行为基线**

Run: `cargo test`
Expected: 全部通过（记录当前用例数，后续必须一致）

- [ ] **Step 2: 用 `materialize` + `stream_to_blob` 替换现有 `materialize`**

把 `src/unified.rs` 里从 `async fn materialize(` 到它的结尾 `}`（函数体包含 staging、请求、校验、blob 落盘）整体替换为：

```rust
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
    validate_relative("repository path", &remote.path)?;
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
    stream_to_blob(remote.backend, &remote.path, response, Some(remote), cache_root, progress).await
}

/// Stream a response into staging, verify it, and move it into the
/// content-addressed blob store.
///
/// `expected` carries the manifest checks for a repository download. A
/// single-file download passes `None` and relies on the response's
/// `Content-Length` instead.
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
        let _ = fs::remove_file(&staging);
        bail!("incomplete download for {path}");
    }
    let sha256 = format!("{:x}", sha256.finalize());
    let git = format!("{:x}", git.finalize());
    if let Some(expected) = expected.and_then(|remote| remote.sha256.as_deref())
        && expected != sha256
    {
        let _ = fs::remove_file(&staging);
        bail!("SHA-256 mismatch for {path}");
    }
    if let Some(expected) = expected.and_then(|remote| remote.git_blob_id.as_deref())
        && expected != git
    {
        let _ = fs::remove_file(&staging);
        bail!("Git blob hash mismatch for {path}");
    }
    let blob = cache_root.join("blobs").join("sha256").join(&sha256);
    if let Some(parent) = blob.parent() {
        fs::create_dir_all(parent)?;
    }
    if blob.exists() {
        fs::remove_file(&staging)?;
    } else if let Err(error) = fs::rename(&staging, &blob) {
        // Another concurrent file may have produced the same content-addressed
        // blob between the existence check and the rename. In that case the
        // already-complete blob wins and this staging file can be discarded.
        if blob.exists() {
            fs::remove_file(&staging)?;
        } else {
            return Err(error.into());
        }
    }
    Ok((blob, sha256, git))
}
```

- [ ] **Step 3: 抽出进度条 helper 并用于 `download_loaded`**

在 `fn snapshot_root` 之前插入：

```rust
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
```

把 `download_loaded` 里：

```rust
    let progress = if progress {
        ProgressBar::new_spinner()
    } else {
        ProgressBar::hidden()
    };
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}")?,
    );
    progress.set_message(format!("{repo_id} • verifying and downloading"));
    progress.enable_steady_tick(std::time::Duration::from_millis(100));
```

替换为：

```rust
    let progress = progress_bar(progress, format!("{repo_id} • verifying and downloading"))?;
```

- [ ] **Step 4: 运行测试确认行为不变**

Run: `cargo test && cargo clippy --all-targets`
Expected: 与 Step 1 相同的通过用例；clippy 无新增 warning

- [ ] **Step 5: Commit**

```bash
git add src/unified.rs
git commit -m "Split stream_to_blob out of materialize for reusable downloads"
```

---

### Task 4: `download_single_file` 与单文件集成测试

**Files:**
- Create: `tests/download.rs`
- Modify: `Cargo.toml`（dev-dependencies）
- Modify: `src/unified.rs`（新增 `download_single_file`、`single_file_urls`、`send_first_success`、`single_file_result`）
- Modify: `src/ops.rs:10-82`（`DownloadOptions`、`download`）
- Test: `tests/download.rs`、`tests/api.rs`

- [ ] **Step 1: 加 `temp_env` dev-dependency**

在 `Cargo.toml` 末尾追加：

```toml
[dev-dependencies]
temp-env = "0.3"
```

（crate 名是 `temp-env`，库名是 `temp_env`；`std::env::set_var` 在 edition 2024 是 unsafe，而本 crate `unsafe_code = "forbid"`，所以用它来在测试里安全地设置环境变量。）

- [ ] **Step 2: 写失败的集成测试**

创建 `tests/download.rs`：

```rust
//! Download tests against a local mock hub.
//!
//! `std::env::set_var` is unsafe in edition 2024 and this crate forbids unsafe,
//! so environment overrides go through `temp_env` inside a process-wide lock.

use modelhub::{Backend, DownloadOptions, DownloadedRepo, RepoKind};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// Serializes tests that mutate environment variables.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "modelhub-download-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

/// Minimal HTTP server that records request paths and serves canned responses.
struct MockHub {
    address: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockHub {
    fn start(routes: Vec<(String, u16, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock hub");
        let address = format!("http://{}", listener.local_addr().expect("mock hub address"));
        let requests: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let routes = Arc::new(routes);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let recorded = Arc::clone(&recorded);
                let routes = Arc::clone(&routes);
                std::thread::spawn(move || serve(stream, routes.as_slice(), &recorded));
            }
        });
        Self { address, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn serve(mut stream: TcpStream, routes: &[(String, u16, Vec<u8>)], recorded: &Mutex<Vec<String>>) {
    let mut request = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let Ok(count) = stream.read(&mut buffer) else {
            return;
        };
        if count == 0 {
            return;
        }
        request.extend_from_slice(&buffer[..count]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    let text = String::from_utf8_lossy(&request);
    let path = text.split_whitespace().nth(1).unwrap_or("/").to_owned();
    recorded.lock().unwrap().push(path.clone());
    for (prefix, status, body) in routes {
        if path.starts_with(prefix.as_str()) {
            respond(&mut stream, *status, body);
            return;
        }
    }
    respond(&mut stream, 404, b"missing");
}

fn respond(stream: &mut TcpStream, status: u16, body: &[u8]) {
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

/// Point every endpoint and native cache override at the test fixtures.
fn with_hub<R>(mock: &MockHub, root: &Path, body: impl FnOnce() -> R) -> R {
    let _guard = env_lock();
    temp_env::with_vars(
        [
            ("HF_ENDPOINT", Some(mock.address.as_str())),
            ("MODELSCOPE_ENDPOINT", Some(mock.address.as_str())),
            (
                "HUGGINGFACE_HUB_CACHE",
                Some(root.join("hf-native").to_str().unwrap()),
            ),
            (
                "MODELSCOPE_CACHE",
                Some(root.join("ms-native").to_str().unwrap()),
            ),
        ],
        body,
    )
}

fn run(options: &DownloadOptions) -> anyhow::Result<DownloadedRepo> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(modelhub::download(options))
}

fn single_file_options(root: &Path, file: &str) -> DownloadOptions {
    let mut options = DownloadOptions::new("acme/demo");
    options.file = Some(file.to_owned());
    options.cache_root = root.join("cache");
    options
}

fn sha256_hex(body: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(body);
    format!("{:x}", hasher.finalize())
}

#[test]
fn cache_hit_returns_without_touching_the_network() {
    let root = temp_root("cache-hit");
    let cached = root.join("cache/datasets/acme--demo/modelscope/snapshots/master/data/a.mp3");
    fs::create_dir_all(cached.parent().unwrap()).unwrap();
    fs::write(&cached, b"audio").unwrap();
    let mock = MockHub::start(Vec::new());

    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "data/a.mp3");
        options.kind = Some(RepoKind::Dataset);
        options.backend = Some(Backend::ModelScope);
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Dataset);
        assert_eq!(downloaded.file.as_deref(), Some(cached.as_path()));
    });

    assert!(mock.requests().is_empty());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn known_kind_and_backend_issue_exactly_one_request() {
    let root = temp_root("one-request");
    let mock = MockHub::start(vec![(
        "/api/v1/datasets/acme--demo/repo?Revision=master&FilePath=data%2Fa.mp3".to_owned(),
        200,
        b"audio".to_vec(),
    )]);

    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "data/a.mp3");
        options.kind = Some(RepoKind::Dataset);
        options.backend = Some(Backend::ModelScope);
        let downloaded = run(&options).unwrap();

        assert_eq!(downloaded.kind, RepoKind::Dataset);
        let expected =
            root.join("cache/datasets/acme--demo/modelscope/snapshots/master/data/a.mp3");
        assert_eq!(downloaded.file.as_deref(), Some(expected.as_path()));
        assert_eq!(fs::read(&expected).unwrap(), b"audio");
        assert!(downloaded.modelscope_root.is_some());
        assert!(downloaded.huggingface_root.is_none());
        assert!(!root.join("cache/datasets/acme--demo/.modelhub-manifest.json").exists());
        let blob = root.join("cache/blobs/sha256").join(sha256_hex(b"audio"));
        assert!(blob.is_file());
    });

    assert_eq!(mock.requests().len(), 1);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn unsafe_paths_fail_before_any_request() {
    let root = temp_root("validate");
    let mock = MockHub::start(Vec::new());

    with_hub(&mock, &root, || {
        for file in ["../evil", "/etc/passwd", "a/../../b"] {
            let options = single_file_options(&root, file);
            assert!(run(&options).is_err(), "{file} must be rejected");
        }
        let mut options = single_file_options(&root, "data/a.mp3");
        options.revision = Some("..".to_owned());
        assert!(run(&options).is_err());
        options.revision = Some("/main".to_owned());
        assert!(run(&options).is_err());
    });

    assert!(mock.requests().is_empty());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn missing_hints_probe_candidates_and_first_success_wins() {
    let root = temp_root("race");
    let mock = MockHub::start(vec![(
        "/models/acme--demo/resolve/main/notes.txt".to_owned(),
        200,
        b"hello".to_vec(),
    )]);

    with_hub(&mock, &root, || {
        let options = single_file_options(&root, "notes.txt");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Model);
        assert!(downloaded.huggingface_root.is_some());
        let file = downloaded.file.as_deref().unwrap();
        assert_eq!(fs::read(file).unwrap(), b"hello");
    });

    let requests = mock.requests();
    assert!(
        requests
            .iter()
            .any(|path| path == "/models/acme--demo/resolve/main/notes.txt")
    );
    assert!(requests.len() <= 4);
    assert!(
        requests
            .iter()
            .all(|path| !path.contains("tree") && !path.contains("blobs=true"))
    );
    fs::remove_dir_all(&root).unwrap();
}
```

在 `tests/api.rs` 的 `async_operations_are_exported` 末尾追加：

```rust
    let mut download = modelhub::DownloadOptions::new("acme/demo");
    download.kind = Some(modelhub::RepoKind::Dataset);
    download.backend = Some(modelhub::Backend::ModelScope);
    assert!(download.file.is_none());
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cargo test --test download`
Expected: 编译失败，`DownloadOptions` 没有 `kind`/`backend` 字段

- [ ] **Step 4: 实现 `download_single_file`**

在 `src/unified.rs` 的 `fn add_separate` 之后插入：

```rust
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
        Backend::ModelScope => vec![modelscope_file_url(kind, repo_id, revision, file)],
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
    let requests = futures_util::stream::iter(candidates.into_iter().map(|(kind, backend, url)| {
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
    validate_relative("file path", file)?;
    if let Some(revision) = revision {
        validate_relative("revision", revision)?;
    }
    let kinds = kind.map_or_else(|| vec![RepoKind::Model, RepoKind::Dataset], |kind| vec![kind]);
    let backends = backend.map_or_else(
        || vec![Backend::HuggingFace, Backend::ModelScope],
        |backend| vec![backend],
    );
    let repo_root = |kind: RepoKind| cache_root.join(kind.segment()).join(repo_id.replace('/', "--"));
    for kind in &kinds {
        for backend in &backends {
            let revision = revision.map_or_else(|| backend.default_revision(), str::to_owned);
            let target = safe_path(&snapshot_root(&repo_root(*kind), *backend, &revision), file)?;
            if target.is_file() {
                return Ok(single_file_result(*kind, *backend, repo_root(*kind), target));
            }
        }
    }
    let mut candidates = Vec::new();
    for kind in &kinds {
        for backend in &backends {
            let revision = revision.map_or_else(|| backend.default_revision(), str::to_owned);
            for url in single_file_urls(*backend, *kind, repo_id, &revision, file) {
                candidates.push((*kind, *backend, url));
            }
        }
    }
    let hf_client = huggingface_client()?;
    let ms_client = crate::modelscope::client::http_client()?;
    let progress = progress_bar(progress, format!("{repo_id} • downloading {file}"))?;
    let (kind, backend, _, response) =
        send_first_success(candidates, file, repo_id, &hf_client, &ms_client).await?;
    let (blob, _, _) =
        match stream_to_blob(backend, file, response, None, cache_root, &progress).await {
            Ok(value) => value,
            Err(error) => {
                progress.abandon_with_message(format!("{repo_id} • download failed"));
                return Err(error);
            }
        };
    let revision = revision.map_or_else(|| backend.default_revision(), str::to_owned);
    let target = safe_path(&snapshot_root(&repo_root(kind), backend, &revision), file)?;
    link_artifact(&blob, &target)?;
    progress.finish_with_message(format!("✓ {repo_id} • downloaded {file}"));
    Ok(single_file_result(kind, backend, repo_root(kind), target))
}
```

- [ ] **Step 5: 给 `DownloadOptions` 加字段并在 `download()` 分派**

`src/ops.rs` 顶部导入改为：

```rust
use crate::unified::{Backend, DownloadedRepo, RepoManifest};
```

`DownloadOptions` 替换为：

```rust
/// Options for [`download`].
#[derive(Clone, Debug)]
pub struct DownloadOptions {
    pub repo_id: String,
    /// Optional single file to download; `None` downloads the whole repository.
    ///
    /// A single file is fetched with one request and never lists the repository.
    pub file: Option<String>,
    /// Restrict the download to a model or a dataset; `None` auto-detects it.
    pub kind: Option<RepoKind>,
    /// Restrict the download to one backend; `None` uses every supported backend.
    pub backend: Option<Backend>,
    /// Revision to request. Defaults to `main`/`master` per backend.
    pub revision: Option<String>,
    /// Root directory owned by modelhub.
    pub cache_root: PathBuf,
    /// Maximum number of files downloaded concurrently.
    pub jobs: usize,
    /// Keep both backend versions even when model weights differ.
    pub all_backends: bool,
    /// Show a progress bar.
    pub progress: bool,
}

impl DownloadOptions {
    /// Options for `repo_id`, defaulting to the modelhub cache and 4 jobs.
    #[must_use]
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            file: None,
            kind: None,
            backend: None,
            revision: None,
            cache_root: crate::cache::cache_dir(),
            jobs: 4,
            all_backends: false,
            progress: false,
        }
    }
}
```

`download()` 替换为：

```rust
/// Download a repository or a single file.
///
/// Auto-detects whether `repo_id` is a model or a dataset and which backends
/// host it unless [`DownloadOptions::kind`] or [`DownloadOptions::backend`] say
/// otherwise. A single-file download is stored in the modelhub cache only and
/// is not linked into the native backend caches.
pub async fn download(opts: &DownloadOptions) -> Result<DownloadedRepo> {
    fs::create_dir_all(&opts.cache_root)?;
    if let Some(file) = opts.file.as_deref() {
        return crate::unified::download_single_file(
            opts.kind,
            opts.backend,
            &opts.repo_id,
            opts.revision.as_deref(),
            file,
            &opts.cache_root,
            opts.progress,
        )
        .await;
    }
    let downloaded = crate::unified::download_repo(
        &opts.repo_id,
        opts.revision.as_deref().unwrap_or("main"),
        opts.revision.as_deref().unwrap_or("master"),
        &opts.cache_root,
        opts.jobs,
        opts.all_backends,
        None,
        opts.progress,
    )
    .await?;
    if let Some(root) = downloaded.huggingface_root.as_deref() {
        link_directory(root, &huggingface_cache_path(downloaded.kind, &opts.repo_id))?;
    }
    if let Some(root) = downloaded.modelscope_root.as_deref() {
        link_directory(root, &modelscope_cache_path(downloaded.kind, &opts.repo_id))?;
    }
    Ok(downloaded)
}
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cargo test --test download && cargo test --test api`
Expected: `tests/download.rs` 4 passed；`tests/api.rs` 3 passed

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock tests/download.rs tests/api.rs src/unified.rs src/ops.rs
git commit -m "Download single files without listing the repository"
```

---

### Task 5: 整仓清单探测按 `kind`/`backend` 收窄

**Files:**
- Modify: `src/unified.rs:642-770`（`download_repo`、`detect_manifests`、`probe_manifests`、`download_loaded`）
- Modify: `src/ops.rs`（`download` 调用 `download_repo` 传 `kind`/`backend`）
- Test: `tests/download.rs`

- [ ] **Step 1: 写失败的集成测试**

在 `tests/download.rs` 末尾追加：

```rust
#[test]
fn whole_repo_with_kind_and_backend_skips_detection() {
    let root = temp_root("whole-hints");
    let mock = MockHub::start(vec![
        (
            "/api/v1/models/acme--demo/repo/files?Recursive=true&Revision=master".to_owned(),
            200,
            br#"{"Success":true,"Data":{"Files":[{"Path":"config.json","Size":7,"Type":"blob"}]}}"#
                .to_vec(),
        ),
        (
            "/api/v1/models/acme--demo/repo?Revision=master&FilePath=config.json".to_owned(),
            200,
            br#"{"a":1}"#.to_vec(),
        ),
    ]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::ModelScope);
        options.cache_root = root.join("cache");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Model);
        assert!(downloaded.huggingface_root.is_none());
        assert!(downloaded.file.is_none());
        let snapshot = root.join("cache/models/acme--demo/modelscope/snapshots/master/config.json");
        assert_eq!(fs::read(&snapshot).unwrap(), br#"{"a":1}"#);
        let manifest =
            fs::read_to_string(root.join("cache/models/acme--demo/.modelhub-manifest.json"))
                .unwrap();
        assert!(manifest.contains("\"modelscope\""));
        assert!(!manifest.contains("\"huggingface\""));
    });

    let requests = mock.requests();
    assert!(requests.iter().all(|path| path.starts_with("/api/v1/models/")));
    assert_eq!(requests.len(), 2);
    assert!(root.join("ms-native/models/acme--demo").exists());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn whole_repo_with_kind_probes_both_backends_for_that_kind_only() {
    let root = temp_root("whole-kind");
    let mock = MockHub::start(vec![
        (
            "/api/datasets/acme--demo/revision/main?blobs=true".to_owned(),
            200,
            br#"{"sha":"abc123","siblings":[]}"#.to_vec(),
        ),
        (
            "/api/v1/datasets/acme--demo/repo/tree?Recursive=True&Revision=master&PageNumber=1&PageSize=200".to_owned(),
            200,
            br#"{"Data":{"Files":[]}}"#.to_vec(),
        ),
    ]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.kind = Some(RepoKind::Dataset);
        options.cache_root = root.join("cache");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Dataset);
        assert!(downloaded.huggingface_root.is_some());
        assert!(downloaded.modelscope_root.is_some());
    });

    let requests = mock.requests();
    assert!(requests.iter().all(|path| path.contains("/datasets/")));
    assert!(!requests.iter().any(|path| path.contains("/models/")));
    assert_eq!(requests.len(), 2);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn whole_repo_with_backend_probes_both_kinds_on_that_backend() {
    let root = temp_root("whole-backend");
    let mock = MockHub::start(vec![(
        "/api/v1/datasets/acme--demo/repo/tree?Recursive=True&Revision=master&PageNumber=1&PageSize=200".to_owned(),
        200,
        br#"{"Data":{"Files":[]}}"#.to_vec(),
    )]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.backend = Some(Backend::ModelScope);
        options.cache_root = root.join("cache");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Dataset);
        assert!(downloaded.modelscope_root.is_some());
        assert!(downloaded.huggingface_root.is_none());
    });

    let requests = mock.requests();
    assert!(requests.iter().all(|path| path.starts_with("/api/v1/")));
    assert_eq!(requests.len(), 2);
    fs::remove_dir_all(&root).unwrap();
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --test download whole_repo`
Expected: 失败 —— `backend: Some(ModelScope)` 目前仍会访问 Hugging Face，断言 `requests.iter().all(...)` 或请求数失败

- [ ] **Step 3: 让 `detect_manifests` / `probe_manifests` 接受提示**

把 `src/unified.rs` 的 `download_repo` 替换为：

```rust
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
```

把 `detect_manifests` 替换为：

```rust
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
    let (model, dataset) = match want_kind {
        Some(RepoKind::Model) => (
            probe_manifests(
                RepoKind::Model,
                want_backend,
                hf_client,
                ms_client,
                repo_id,
                huggingface_revision,
                modelscope_revision,
            )
            .await,
            (None, None),
        ),
        Some(RepoKind::Dataset) => (
            (None, None),
            probe_manifests(
                RepoKind::Dataset,
                want_backend,
                hf_client,
                ms_client,
                repo_id,
                huggingface_revision,
                modelscope_revision,
            )
            .await,
        ),
        None => tokio::join!(
            probe_manifests(
                RepoKind::Model,
                want_backend,
                hf_client,
                ms_client,
                repo_id,
                huggingface_revision,
                modelscope_revision
            ),
            probe_manifests(
                RepoKind::Dataset,
                want_backend,
                hf_client,
                ms_client,
                repo_id,
                huggingface_revision,
                modelscope_revision
            )
        ),
    };
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
```

把 `probe_manifests` 替换为：

```rust
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
```

- [ ] **Step 4: 从 `download_loaded` 移除单文件分支**

删除 `download_loaded` 签名里的 `file: Option<&str>,` 参数（保留 `progress: bool`）。

删除函数体里这段：

```rust
    let file = file.map(|file| file.trim_start_matches('/').to_owned());
    if let Some(file) = file.as_deref() {
        if !paths.contains(file) {
            bail!("`{file}` is not present in {} `{repo_id}`", kind.label());
        }
        paths.retain(|path| path == file);
    }
```

删除 `let file_path = ...` 整段（从 `let file_path = match file.as_deref() {` 到 `};`），最后的结果改为：

```rust
    Ok(DownloadedRepo {
        kind,
        repo_root: repo_root.clone(),
        huggingface_root: hf_root,
        modelscope_root: ms_root,
        file: None,
    })
```

把：

```rust
    // Only a full download yields a complete manifest; a single-file download
    // stays unverifiable and reports `unknown` in `list --check`.
    if file.is_none() {
        write_repo_manifest(&repo_root, kind, hf.as_ref(), ms.as_ref())?;
    }
```

替换为：

```rust
    write_repo_manifest(&repo_root, kind, hf.as_ref(), ms.as_ref())?;
```

- [ ] **Step 5: 让 `ops::download` 传递提示**

把 `src/ops.rs` 里 `download_repo(...)` 调用的参数列表从：

```rust
        opts.jobs,
        opts.all_backends,
        None,
        opts.progress,
    )
```

替换为：

```rust
        opts.jobs,
        opts.all_backends,
        opts.progress,
        opts.kind,
        opts.backend,
    )
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cargo test`
Expected: 全部通过（含 `tests/download.rs` 7 passed）

- [ ] **Step 7: Commit**

```bash
git add src/unified.rs src/ops.rs tests/download.rs
git commit -m "Narrow repository manifest probing with kind and backend hints"
```

---

### Task 6: CLI `--repo-type` / `--backend`

**Files:**
- Modify: `src/main.rs:4`、`19-39`、`126-162`、`294-320`、`578-586`

- [ ] **Step 1: 写失败的 CLI 解析测试**

把 `src/main.rs` 测试模块里的 `download_accepts_an_optional_file` 替换为：

```rust
    #[test]
    fn download_accepts_an_optional_file() {
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "README.md"]).is_ok());
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "data/train.parquet"])
                .is_ok()
        );
    }

    #[test]
    fn download_accepts_kind_and_backend_hints() {
        assert!(
            Cli::try_parse_from([
                "modelhub",
                "download",
                "acme/demo",
                "README.md",
                "--repo-type",
                "dataset",
                "--backend",
                "modelscope",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "--repo-type", "model"])
                .is_ok()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "--backend", "huggingface"])
                .is_ok()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "--backend", "modelhub"])
                .is_err()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "--repo-type", "nope"])
                .is_err()
        );
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --bin modelhub download_accepts_kind_and_backend_hints`
Expected: 失败，`unexpected argument '--repo-type'`

- [ ] **Step 3: 加 CLI 字段与解析函数**

`src/main.rs` 顶部导入改为：

```rust
use modelhub::{Backend, CacheSource, RepoEntry, RepoKind, RepoStatus, UploadBackend};
```

`Command::Download` 的定义改为：

```rust
    /// Detect the repository kind, download it once, and link all caches.
    Download {
        /// Model or dataset identifier, for example `org/name`. The kind is detected automatically.
        repo_id: String,
        /// Optional single file to download, for example `README.md` or `data/train.parquet`.
        /// Without it the whole repository is downloaded. A single file is fetched directly
        /// without listing the repository.
        file: Option<String>,
        /// Revision to request. Defaults to `master` on `ModelScope` and `main` on Hugging Face.
        #[arg(short, long)]
        revision: Option<String>,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Maximum number of files downloaded concurrently.
        #[arg(short = 'j', long, default_value_t = 4, value_parser = parse_jobs)]
        jobs: usize,
        /// Keep both backend versions even when model weights differ.
        #[arg(long)]
        all_backends: bool,
        /// Repo type when known: `model` or `dataset`. Skips kind detection.
        #[arg(long, value_parser = parse_repo_type)]
        repo_type: Option<RepoKind>,
        /// Backend when known: `huggingface` or `modelscope`. Skips backend probing.
        #[arg(long, value_parser = parse_download_backend)]
        backend: Option<Backend>,
    },
```

在 `parse_repo_type` 之后加：

```rust
/// Parse `--backend` for downloads.
fn parse_download_backend(value: &str) -> Result<Backend, String> {
    match value {
        "huggingface" => Ok(Backend::HuggingFace),
        "modelscope" => Ok(Backend::ModelScope),
        _ => Err("backend must be modelscope or huggingface".to_owned()),
    }
}
```

`main()` 里 `Command::Download { ... }` 的匹配与选项装配改为：

```rust
        Command::Download {
            repo_id,
            file,
            revision,
            cache_dir,
            jobs,
            all_backends,
            repo_type,
            backend,
        } => {
            let mut options = modelhub::DownloadOptions::new(repo_id);
            options.file = file;
            options.kind = repo_type;
            options.backend = backend;
            options.revision = revision;
            if let Some(cache_dir) = cache_dir {
                options.cache_root = cache_dir;
            }
            options.jobs = jobs;
            options.all_backends = all_backends;
            options.progress = true;
            let downloaded = runtime()?.block_on(modelhub::download(&options))?;
            if options.file.is_none() {
                eprintln!("Verified backend manifests and linked compatible cache snapshots.");
            } else {
                let path = downloaded
                    .file
                    .as_deref()
                    .unwrap_or(downloaded.repo_root.as_path());
                println!("Downloaded {} {}", downloaded.kind.label(), path.display());
            }
            Ok(())
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cargo test --bin modelhub && cargo clippy --all-targets`
Expected: 全部通过，clippy 无新增 warning

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "Add --repo-type and --backend hints to the download command"
```

---

### Task 7: README 与全量验证

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新单文件下载说明**

把 README 的：

```markdown
只下载仓库里的单个子文件（路径相对仓库根目录，可含子目录）：

```bash
modelhub download org/name README.md
modelhub download org/name data/train.parquet
```

单文件下载只写入 modelhub 缓存并打印落盘路径，不会链接到 ModelScope / Hugging Face 原生缓存（避免后端把不完整的快照当成整库）。
```

替换为：

```markdown
只下载仓库里的单个子文件（路径相对仓库根目录，可含子目录）：

```bash
modelhub download org/name README.md
modelhub download org/name data/train.parquet
```

单文件下载只请求这一个文件，不会列举整个仓库：命中缓存时直接返回本地路径，不做任何网络请求；未命中时只发这一个文件的请求（最多在 model / dataset 与 ModelScope / Hugging Face 的候选里并行探测，先成功者赢）。只写入 modelhub 缓存并打印落盘路径，不会链接到 ModelScope / Hugging Face 原生缓存（避免后端把不完整的快照当成整库）。

已知仓库类型和后端时可以跳过探测：

```bash
modelhub download org/name data/train.parquet --repo-type dataset --backend modelscope
```

`--repo-type` 与 `--backend` 同样可以用于整仓下载：只传其一时，只探测剩下那一个维度；都传时完全不探测。两个参数都可省略。
```

- [ ] **Step 2: 更新库 API 说明**

把 README 库 API 段落里的：

```rust
// 下载(需 async 运行时)
let mut options = DownloadOptions::new("org/name");
options.file = Some("README.md".to_owned());
let downloaded = runtime.block_on(modelhub::download(&options))?;
```

替换为：

```rust
// 下载(需 async 运行时)
let mut options = DownloadOptions::new("org/name");
options.file = Some("README.md".to_owned());
// 已知类型/后端时跳过探测；省略则自动识别
options.kind = Some(modelhub::RepoKind::Dataset);
options.backend = Some(modelhub::Backend::ModelScope);
let downloaded = runtime.block_on(modelhub::download(&options))?;
```

- [ ] **Step 3: 全量验证**

Run: `cargo fmt --check && cargo clippy --all-targets && cargo test && cargo doc --no-deps`
Expected: 全部通过；`cargo doc` 无 warning

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document single-file download fast path and probe hints"
```

---

## 完成标准

- `cargo fmt --check`、`cargo clippy --all-targets`、`cargo test`、`cargo doc --no-deps` 全绿。
- `download()` 在 `file` 有值时只访问 `kind` 指定的类型；缓存命中零网络；未命中只请求这一个文件，不出现 `repo/tree` 或 `?blobs=true`。
- 单文件不写 `.modelhub-manifest.json`、不链接原生缓存。
- 非法 `file`/`revision` 在任何请求之前报错。
- 整仓下载默认行为不变；`kind`/`backend` 只在显式给出时收窄。
