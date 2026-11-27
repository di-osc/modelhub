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
        Self::start_with(move |stream, _, path| {
            for (prefix, status, body) in &routes {
                if path.starts_with(prefix.as_str()) {
                    respond(stream, *status, body);
                    return;
                }
            }
            respond(stream, 404, b"missing");
        })
    }

    fn start_with(handler: impl Fn(&mut TcpStream, &str, &str) + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock hub");
        let address = format!(
            "http://{}",
            listener.local_addr().expect("mock hub address")
        );
        let requests: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let handler = Arc::new(handler);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let recorded = Arc::clone(&recorded);
                let handler = Arc::clone(&handler);
                std::thread::spawn(move || serve(stream, &*handler, &recorded));
            }
        });
        Self { address, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn serve(
    mut stream: TcpStream,
    handler: impl FnOnce(&mut TcpStream, &str, &str),
    recorded: &Mutex<Vec<String>>,
) {
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
    handler(&mut stream, &text, &path);
}

fn respond(stream: &mut TcpStream, status: u16, body: &[u8]) {
    respond_with(stream, status, body.len(), "", body);
}

fn respond_with(stream: &mut TcpStream, status: u16, length: usize, headers: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status} Result\r\nContent-Length: {length}\r\nConnection: close\r\n{headers}\r\n"
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

    assert_eq!(mock.requests().len(), 0);
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

    assert_eq!(mock.requests().len(), 0);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn unsafe_huggingface_ref_is_ignored() {
    let root = temp_root("bad-ref");
    let repo = root.join("cache/models/acme--demo/huggingface");
    fs::create_dir_all(repo.join("refs")).unwrap();
    // Without the guard, `snapshots/../../../escape` resolves outside the
    // repository root and this file would be returned with no network request.
    fs::write(repo.join("refs/main"), b"../../../escape").unwrap();
    // `snapshots/` must exist for the `..` components to resolve.
    fs::create_dir_all(repo.join("snapshots")).unwrap();
    let escaped = root.join("cache/models/escape/config.json");
    fs::create_dir_all(escaped.parent().unwrap()).unwrap();
    fs::write(&escaped, b"{}").unwrap();
    let mock = MockHub::start(Vec::new());

    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "config.json");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::HuggingFace);
        // The unsafe ref is ignored, so the request falls through to the
        // network and fails against the mock instead of returning the escape.
        assert!(run(&options).is_err());
    });

    assert_ne!(mock.requests().len(), 0);
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
        // A single-file download never links into the native caches, so neither
        // native cache root is even created.
        assert!(!root.join("ms-native").exists());
        assert!(!root.join("hf-native").exists());
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
        let mut options = single_file_options(&root, "data/a.mp3");
        options.repo_id = "../evil".to_owned();
        assert!(run(&options).is_err());
    });

    assert_eq!(mock.requests().len(), 0);
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

#[test]
fn whole_repo_with_kind_and_backend_skips_detection() {
    let root = temp_root("whole-hints");
    let mock = MockHub::start(vec![
        (
            "/api/v1/models/acme/demo/repo/files?Recursive=true&Revision=master".to_owned(),
            200,
            br#"{"Success":true,"Data":{"Files":[{"Path":"config.json","Size":7,"Type":"blob"}]}}"#
                .to_vec(),
        ),
        (
            "/api/v1/models/acme/demo/repo?Revision=master&FilePath=config.json".to_owned(),
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
    assert!(
        requests
            .iter()
            .all(|path| path.starts_with("/api/v1/models/"))
    );
    assert_eq!(requests.len(), 2);
    assert!(root.join("ms-native/models/acme--demo").exists());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn whole_repo_with_kind_probes_both_backends_for_that_kind_only() {
    let root = temp_root("whole-kind");
    let mock = MockHub::start(vec![
        (
            "/api/datasets/acme/demo/revision/main?blobs=true".to_owned(),
            200,
            br#"{"sha":"abc123","siblings":[]}"#.to_vec(),
        ),
        (
            "/api/v1/datasets/acme/demo/repo/tree?Recursive=True&Revision=master&PageNumber=1&PageSize=200".to_owned(),
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
        "/api/v1/datasets/acme/demo/repo/tree?Recursive=True&Revision=master&PageNumber=1&PageSize=200".to_owned(),
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

#[test]
fn whole_repo_without_hints_stops_after_a_model_hit() {
    let root = temp_root("whole-default");
    let mock = MockHub::start(vec![
        (
            "/api/models/acme/demo/revision/main?blobs=true".to_owned(),
            200,
            br#"{"sha":"abc123","siblings":[{"rfilename":"config.json","size":7}]}"#.to_vec(),
        ),
        (
            "/acme/demo/resolve/abc123/config.json".to_owned(),
            200,
            br#"{"a":1}"#.to_vec(),
        ),
    ]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.cache_root = root.join("cache");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Model);
        assert!(downloaded.file.is_none());
        let snapshot =
            root.join("cache/models/acme--demo/huggingface/snapshots/abc123/config.json");
        assert_eq!(fs::read(&snapshot).unwrap(), br#"{"a":1}"#);
    });

    let requests = mock.requests();
    assert!(
        requests
            .iter()
            .any(|path| path == "/api/models/acme/demo/revision/main?blobs=true")
    );
    // Model probes on both backends, then one file. The dataset is never probed
    // once the model is found.
    assert!(
        !requests.iter().any(|path| path.contains("/datasets/")),
        "{requests:?}"
    );
    assert_eq!(requests.len(), 3);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn whole_repo_prefers_the_model_when_both_exist() {
    let root = temp_root("whole-ambiguous");
    let mock = MockHub::start(vec![
        (
            "/api/models/acme/demo/revision/main?blobs=true".to_owned(),
            200,
            br#"{"sha":"abc","siblings":[]}"#.to_vec(),
        ),
        (
            "/api/v1/datasets/acme/demo/repo/tree?Recursive=True&Revision=master&PageNumber=1&PageSize=200".to_owned(),
            200,
            br#"{"Data":{"Files":[]}}"#.to_vec(),
        ),
    ]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.cache_root = root.join("cache");
        let downloaded = run(&options).unwrap();
        assert_eq!(downloaded.kind, RepoKind::Model);
    });

    // The model hit ends detection, so the dataset route is never requested.
    let requests = mock.requests();
    assert!(
        !requests.iter().any(|path| path.contains("/datasets/")),
        "{requests:?}"
    );
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn unsafe_whole_repo_revision_fails_before_any_request() {
    let root = temp_root("whole-revision");
    let mock = MockHub::start(Vec::new());

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::ModelScope);
        options.revision = Some("../../../escape".to_owned());
        options.cache_root = root.join("cache");
        assert!(run(&options).is_err());
    });

    assert_eq!(mock.requests().len(), 0);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn unsafe_huggingface_manifest_revision_is_rejected() {
    let root = temp_root("hf-bad-sha");
    let mock = MockHub::start(vec![(
        "/api/models/acme/demo/revision/main?blobs=true".to_owned(),
        200,
        br#"{"sha":"../../../evil","siblings":[{"rfilename":"config.json","size":7}]}"#.to_vec(),
    )]);

    with_hub(&mock, &root, || {
        let mut options = DownloadOptions::new("acme/demo");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::HuggingFace);
        options.cache_root = root.join("cache");
        assert!(run(&options).is_err());
    });

    // The unsafe manifest revision must never become a file request.
    let requests = mock.requests();
    assert!(
        requests.iter().all(|path| !path.contains("evil")),
        "{requests:?}"
    );
    fs::remove_dir_all(&root).unwrap();
}

fn request_header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
    request.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

#[test]
fn interrupted_single_file_resumes_or_restarts_safely() {
    const OLD_TAG: &str = "ETag: \"v1\"\r\n";
    const NEW_TAG: &str = "ETag: \"v2\"\r\n";
    let cases = [
        (
            206,
            "Content-Range: bytes 3-5/6\r\n",
            &b"def"[..],
            &b"abcdef"[..],
            false,
        ),
        (200, NEW_TAG, &b"uvwxyz"[..], &b"uvwxyz"[..], false),
        (
            206,
            "Content-Range: bytes 2-5/6\r\n",
            &b"wxyz"[..],
            &b"uvwxyz"[..],
            true,
        ),
        (
            206,
            "Content-Range: bytes 3-5/6\r\nETag: \"v2\"\r\n",
            &b"xyz"[..],
            &b"uvwxyz"[..],
            true,
        ),
        (
            416,
            "Content-Range: bytes */3\r\n",
            &b""[..],
            &b"uvwxyz"[..],
            true,
        ),
    ];

    for (status, headers, range_body, expected, restarts) in cases {
        let root = temp_root("single-resume");
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&attempts);
        let mock = MockHub::start_with(move |stream, request, _| {
            let range = request_header(request, "range");
            let if_range = request_header(request, "if-range");
            let attempt = {
                let mut seen = seen.lock().unwrap();
                seen.push((range.map(str::to_owned), if_range.map(str::to_owned)));
                seen.len()
            };
            match attempt {
                // Announce six bytes and close the connection after three.
                1 => respond_with(stream, 200, 6, OLD_TAG, b"abc"),
                2 => respond_with(stream, status, range_body.len(), headers, range_body),
                _ => respond_with(stream, 200, 6, NEW_TAG, b"uvwxyz"),
            }
        });
        with_hub(&mock, &root, || {
            let mut options = single_file_options(&root, "weights.bin");
            options.kind = Some(RepoKind::Model);
            options.backend = Some(Backend::HuggingFace);
            assert!(run(&options).is_err());
            let snapshot =
                root.join("cache/models/acme--demo/huggingface/snapshots/main/weights.bin");
            assert!(!snapshot.exists());

            let result = run(&options).unwrap();
            assert_eq!(result.file.as_deref(), Some(snapshot.as_path()));
            assert_eq!(fs::read(&snapshot).unwrap(), expected);
            assert_eq!(run(&options).unwrap().file, result.file);
        });

        let mut expected_requests = vec![
            (None, None),
            (Some("bytes=3-".to_owned()), Some("\"v1\"".to_owned())),
        ];
        if restarts {
            expected_requests.push((None, None));
        }
        assert_eq!(*attempts.lock().unwrap(), expected_requests);
        fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn interrupted_whole_repo_validates_resumed_bytes_against_the_manifest() {
    for prefix in [b"abc", b"old"] {
        let root = temp_root("whole-resume");
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&attempts);
        let checksum = sha256_hex(b"abcdef");
        let mock = MockHub::start_with(move |stream, request, path| {
            if path.contains("/repo/files?") {
                let manifest = format!(
                    r#"{{"Success":true,"Data":{{"Files":[{{"Path":"weights.bin","Size":6,"Type":"blob","Sha256":"{checksum}"}}]}}}}"#
                );
                respond(stream, 200, manifest.as_bytes());
                return;
            }
            let range = request_header(request, "range").map(str::to_owned);
            let attempt = {
                let mut seen = seen.lock().unwrap();
                seen.push(range);
                seen.len()
            };
            match attempt {
                1 => respond_with(stream, 200, 6, "", prefix),
                2 => respond_with(stream, 206, 3, "Content-Range: bytes 3-5/6\r\n", b"def"),
                _ => respond(stream, 200, b"abcdef"),
            }
        });
        with_hub(&mock, &root, || {
            let mut options = DownloadOptions::new("acme/demo");
            options.cache_root = root.join("cache");
            options.kind = Some(RepoKind::Model);
            options.backend = Some(Backend::ModelScope);
            assert!(run(&options).is_err());
            let snapshot =
                root.join("cache/models/acme--demo/modelscope/snapshots/master/weights.bin");
            assert!(!snapshot.exists());

            assert!(run(&options).unwrap().modelscope_root.is_some());
            assert_eq!(fs::read(&snapshot).unwrap(), b"abcdef");
            assert_eq!(
                fs::read(root.join("ms-native/models/acme--demo/snapshots/master/weights.bin"))
                    .unwrap(),
                b"abcdef"
            );

            // A complete blob still satisfies a later whole-repo download.
            run(&options).unwrap();
        });

        let mut expected_requests = vec![None, Some("bytes=3-".to_owned())];
        if prefix == b"old" {
            expected_requests.push(None);
        }
        assert_eq!(*attempts.lock().unwrap(), expected_requests);
        fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn interrupted_single_file_without_a_strong_etag_restarts() {
    let root = temp_root("weak-etag");
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&attempts);
    let mock = MockHub::start_with(move |stream, request, _| {
        let mut seen = seen.lock().unwrap();
        seen.push(request_header(request, "range").map(str::to_owned));
        if seen.len() == 1 {
            respond_with(stream, 200, 6, "ETag: W/\"v1\"\r\n", b"abc");
        } else {
            respond(stream, 200, b"uvwxyz");
        }
    });
    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "weights.bin");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::ModelScope);
        assert!(run(&options).is_err());
        let file = run(&options).unwrap().file.unwrap();
        assert_eq!(fs::read(file).unwrap(), b"uvwxyz");
    });
    assert_eq!(*attempts.lock().unwrap(), [None, None]);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn interrupted_huggingface_download_resumes_across_a_redirect() {
    // `hf-mirror.com` answers `resolve` requests with a redirect to the CDN,
    // so the resume headers have to survive that hop.
    let root = temp_root("redirect-resume");
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&attempts);
    let mock = MockHub::start_with(move |stream, request, path| {
        if path.starts_with("/cdn/") {
            let mut seen = seen.lock().unwrap();
            seen.push((
                request_header(request, "range").map(str::to_owned),
                request_header(request, "if-range").map(str::to_owned),
            ));
            match seen.len() {
                1 => respond_with(stream, 200, 6, "ETag: \"v1\"\r\n", b"abc"),
                _ => respond_with(
                    stream,
                    206,
                    3,
                    "ETag: \"v1\"\r\nContent-Range: bytes 3-5/6\r\n",
                    b"def",
                ),
            }
            return;
        }
        respond_with(
            stream,
            302,
            0,
            "Location: /cdn/weights.bin\r\nETag: \"v1\"\r\n",
            b"",
        );
    });
    with_hub(&mock, &root, || {
        let mut options = single_file_options(&root, "weights.bin");
        options.kind = Some(RepoKind::Model);
        options.backend = Some(Backend::HuggingFace);
        assert!(run(&options).is_err());
        let file = run(&options).unwrap().file.unwrap();
        assert_eq!(fs::read(file).unwrap(), b"abcdef");
    });
    assert_eq!(
        *attempts.lock().unwrap(),
        [
            (None, None),
            (Some("bytes=3-".to_owned()), Some("\"v1\"".to_owned())),
        ]
    );
    fs::remove_dir_all(&root).unwrap();
}
