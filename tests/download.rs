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
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        let address = format!(
            "http://{}",
            listener.local_addr().expect("mock hub address")
        );
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
fn cache_hit_via_huggingface_ref_skips_network() {
    let root = temp_root("ref-hit");
    let repo = root.join("cache/models/acme--demo/huggingface");
    let snapshot = repo.join("snapshots/abc123");
    fs::create_dir_all(&snapshot).unwrap();
    let cached = snapshot.join("config.json");
    fs::write(&cached, b"{}").unwrap();
    fs::create_dir_all(repo.join("refs")).unwrap();
    fs::write(repo.join("refs/main"), b"abc123").unwrap();
    let mock = MockHub::start(Vec::new());

    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "config.json");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::HuggingFace);
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Model);
        assert_eq!(downloaded.file.as_deref(), Some(cached.as_path()));
    });

    assert!(mock.requests().is_empty());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn known_kind_and_backend_issue_exactly_one_request() {
    let root = temp_root("one-request");
    let mock = MockHub::start(vec![(
        "/api/v1/datasets/acme/demo/repo?Revision=master&FilePath=data%2Fa.mp3".to_owned(),
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
        assert!(
            !root
                .join("cache/datasets/acme--demo/.modelhub-manifest.json")
                .exists()
        );
        let repo = root.join("cache/datasets/acme--demo");
        assert_eq!(
            fs::read_to_string(repo.join(".modelhub-model-id")).unwrap(),
            "acme/demo"
        );
        assert!(repo.join(".modelhub-layout").is_file());
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
        "/acme/demo/resolve/main/notes.txt".to_owned(),
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
            .any(|path| path == "/acme/demo/resolve/main/notes.txt")
    );
    assert!(requests.len() <= 4);
    assert!(
        requests
            .iter()
            .all(|path| !path.contains("tree") && !path.contains("blobs=true"))
    );
    fs::remove_dir_all(&root).unwrap();
}
