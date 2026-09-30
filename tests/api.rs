//! Exercises the public API as an external consumer would.

use modelhub::{CacheSource, ClearOptions, ListOptions, RepoEntry, RepoStatus, clear, list};
use std::fs;
use std::path::{Path, PathBuf};

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "modelhub-api-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

/// A modelhub-owned model with a manifest that matches its snapshot.
fn write_fixture(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let modelhub_root = root.join("modelhub");
    let modelscope_cache = root.join("modelscope");
    let huggingface_hub = root.join("huggingface");
    let repo = modelhub_root.join("models").join("acme--demo");
    let snapshot = repo.join("huggingface").join("snapshots").join("abc");
    fs::create_dir_all(&snapshot).unwrap();
    fs::write(snapshot.join("config.json"), b"{}").unwrap();
    fs::write(
        repo.join(".modelhub-manifest.json"),
        r#"{"version":1,"kind":"model","backends":{"huggingface":{"revision":"abc","files":{"config.json":2}}}}"#,
    )
    .unwrap();
    (modelhub_root, modelscope_cache, huggingface_hub)
}

#[test]
fn list_reports_status_through_the_public_api() {
    let root = temp_root("list");
    let (modelhub_root, modelscope_cache, huggingface_hub) = write_fixture(&root);

    let options = ListOptions {
        cache_root: modelhub_root.clone(),
        check: true,
        modelscope_cache: Some(modelscope_cache),
        huggingface_hub: Some(huggingface_hub),
    };
    let entries: Vec<RepoEntry> = list(&options).unwrap();
    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(entry.id, "acme/demo");
    assert_eq!(entry.status, Some(RepoStatus::Complete));
    assert_eq!(entry.sources, vec![CacheSource::ModelHub]);

    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn clear_reports_removals_through_the_public_api() {
    let root = temp_root("clear");
    let (modelhub_root, modelscope_cache, huggingface_hub) = write_fixture(&root);

    let options = ClearOptions {
        repo_id: Some("acme/demo".to_owned()),
        all: false,
        backend: None,
        cache_root: modelhub_root.clone(),
        modelscope_cache: Some(modelscope_cache),
        huggingface_hub: Some(huggingface_hub),
    };
    let summary = clear(&options).unwrap();
    assert!(summary.found);
    assert_eq!(summary.targets, vec![CacheSource::ModelHub]);
    assert!(!summary.removed.is_empty());
    assert!(!modelhub_root.join("models").join("acme--demo").exists());

    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn async_operations_are_exported() {
    // Referencing the async entry points keeps the public surface honest.
    let _ = modelhub::download;
    let _ = modelhub::check;
    let _ = modelhub::upload;
    let _options = modelhub::DownloadOptions::new("acme/demo");
    let _check_options = modelhub::CheckOptions::default();
    let options = modelhub::UploadOptions::new(
        "acme/demo",
        modelhub::RepoKind::Dataset,
        vec!["/tmp".into()],
    );
    assert!(options.create);
    assert_eq!(options.kind, modelhub::RepoKind::Dataset);
    assert!(options.backends.is_empty());

    let mut download = modelhub::DownloadOptions::new("acme/demo");
    download.kind = Some(modelhub::RepoKind::Dataset);
    download.backend = Some(modelhub::Backend::ModelScope);
    assert!(download.file.is_none());
}
