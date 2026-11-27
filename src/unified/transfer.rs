//! HTTP transfers and on-disk state for interrupted downloads.

use super::{Backend, RemoteFile, hash_contents, hashers};
use anyhow::{Context, bail};
use indicatif::ProgressBar;
use reqwest::{RequestBuilder, Response, StatusCode, header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Deserialize, Serialize)]
struct Partial {
    size: Option<u64>,
    etag: Option<String>,
}

struct Staging {
    path: PathBuf,
    // Keep the state file in place so every process locks the same file, even
    // after the download publishes or discards the corresponding .part.
    state: Option<File>,
    keep: bool,
}

impl Staging {
    fn open(
        cache_root: &Path,
        backend: Backend,
        url: &str,
        expected: Option<&RemoteFile>,
    ) -> anyhow::Result<Self> {
        let dir = cache_root
            .join("staging")
            .join(backend.segment())
            .join("resume");
        fs::create_dir_all(&dir)?;
        let mut key = Sha256::new();
        key.update(url);
        if let Some(remote) = expected {
            key.update(b"\0");
            key.update(remote.size.to_be_bytes());
            key.update(remote.sha256.as_deref().unwrap_or_default());
            key.update(b"\0");
            key.update(remote.git_blob_id.as_deref().unwrap_or_default());
        }
        let key = format!("{:x}", key.finalize());
        let state = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(format!("{key}.state")))?;
        if state.try_lock().is_ok() {
            return Ok(Self {
                path: dir.join(format!("{key}.part")),
                state: Some(state),
                keep: true,
            });
        }

        // A concurrent process owns the partial, or the filesystem does not
        // support file locking. Use an independent staging file in either case.
        let pid = std::process::id();
        let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            path: dir.join(format!("{key}.{pid}.{counter}.part")),
            state: None,
            keep: false,
        })
    }

    fn partial(&mut self, expected: Option<&RemoteFile>) -> Option<(u64, Partial)> {
        let state = self.state.as_mut()?;
        let offset = fs::metadata(&self.path).ok()?.len();
        let partial: Partial = serde_json::from_reader(state).ok()?;
        if offset == 0
            || partial.size.is_some_and(|size| offset >= size)
            || (partial.etag.is_none() && !has_checksum(expected))
        {
            return None;
        }
        Some((offset, partial))
    }

    fn writer(
        &mut self,
        offset: u64,
        partial: Option<&Partial>,
    ) -> anyhow::Result<BufWriter<File>> {
        if offset > 0 {
            return Ok(BufWriter::new(
                OpenOptions::new().append(true).open(&self.path)?,
            ));
        }

        self.keep = false;
        if let Some(state) = &mut self.state {
            // Invalidate the previous validator before replacing the bytes.
            state.set_len(0)?;
            state.rewind()?;
        }
        let file = File::create(&self.path)?;
        if let (Some(state), Some(partial)) = (&mut self.state, partial) {
            serde_json::to_writer(state, partial)?;
            self.keep = true;
        }
        Ok(BufWriter::new(file))
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(super) struct Transfer<'a> {
    response: Response,
    staging: Staging,
    expected: Option<&'a RemoteFile>,
    size: Option<u64>,
    offset: u64,
    retry: Option<RequestBuilder>,
}

impl<'a> Transfer<'a> {
    pub(super) async fn start(
        request: RequestBuilder,
        backend: Backend,
        url: &str,
        cache_root: &Path,
        expected: Option<&'a RemoteFile>,
    ) -> anyhow::Result<Self> {
        let request = request.header(header::ACCEPT_ENCODING, "identity");
        let mut staging = Staging::open(cache_root, backend, url, expected)?;
        if let Some((offset, partial)) = staging.partial(expected)
            && let Some(mut range) = request.try_clone()
        {
            range = range.header(header::RANGE, format!("bytes={offset}-"));
            if let Some(etag) = &partial.etag {
                range = range.header(header::IF_RANGE, etag);
            }
            if let Ok(response) = range.send().await {
                if response.status() == StatusCode::OK {
                    // The server ignored Range or the validator no longer matches.
                    return Self::full(response, staging, expected);
                }
                if let Some(size) = remaining_size(&response, offset, &partial) {
                    return Ok(Self {
                        response,
                        staging,
                        expected,
                        size: Some(size),
                        offset,
                        retry: Some(request),
                    });
                }
            }
            // Some servers reject Range, and a malformed partial response
            // cannot be appended. Try the same endpoint with a normal GET.
        }
        Self::full(request.send().await?, staging, expected)
    }

    fn full(
        response: Response,
        staging: Staging,
        expected: Option<&'a RemoteFile>,
    ) -> anyhow::Result<Self> {
        let status = response.status();
        if !status.is_success() || status == StatusCode::PARTIAL_CONTENT {
            bail!("HTTP {status}");
        }
        let size = expected
            .map(|remote| remote.size)
            .filter(|size| *size > 0)
            .or(response.content_length());
        Ok(Self {
            response,
            staging,
            expected,
            size,
            offset: 0,
            retry: None,
        })
    }

    pub(super) fn size(&self) -> Option<u64> {
        self.size
    }

    pub(super) async fn finish(
        mut self,
        path: &str,
        cache_root: &Path,
        progress: &ProgressBar,
    ) -> anyhow::Result<(PathBuf, String, String)> {
        loop {
            let (sha256, git) = self.write(path, progress).await?;
            let mismatch = self.expected.and_then(|expected| {
                if expected.sha256.as_ref().is_some_and(|hash| hash != &sha256) {
                    Some("SHA-256")
                } else if expected
                    .git_blob_id
                    .as_ref()
                    .is_some_and(|hash| hash != &git)
                {
                    Some("Git blob hash")
                } else {
                    None
                }
            });
            if let Some(checksum) = mismatch {
                self.staging.keep = false;
                if let Some(request) = self.retry.take() {
                    progress.dec(self.size.unwrap_or_default());
                    self = Self::full(request.send().await?, self.staging, self.expected)?;
                    continue;
                }
                bail!("{checksum} mismatch for {path}");
            }

            return self.publish(cache_root, sha256, git);
        }
    }

    async fn write(
        &mut self,
        path: &str,
        progress: &ProgressBar,
    ) -> anyhow::Result<(String, String)> {
        let size = self.size.unwrap_or_default();
        let (mut sha256, mut git) = if self.offset == 0 {
            hashers(size)
        } else {
            let prefix = File::open(&self.staging.path)?.take(self.offset);
            tokio::task::spawn_blocking(move || {
                let (mut sha256, mut git) = hashers(size);
                hash_contents(prefix, &mut sha256, &mut git)?;
                Ok::<_, anyhow::Error>((sha256, git))
            })
            .await??
        };
        let mut written = self.offset;
        let partial = Partial {
            size: self.size,
            etag: strong_etag(&self.response).map(str::to_owned),
        };
        // A response without a usable validator can only be resumed if its
        // complete contents will be checked against a manifest digest.
        let resumable =
            is_identity(&self.response) && (partial.etag.is_some() || has_checksum(self.expected));
        let mut writer = self
            .staging
            .writer(self.offset, resumable.then_some(&partial))?;
        progress.inc(self.offset);
        while let Some(chunk) = self.response.chunk().await? {
            written += chunk.len() as u64;
            if self.size.is_some_and(|size| written > size) {
                self.staging.keep = false;
                bail!("response exceeded the expected size for {path}");
            }
            writer.write_all(&chunk)?;
            sha256.update(&chunk);
            git.update(&chunk);
            progress.inc(chunk.len() as u64);
        }
        writer.flush()?;
        if self.size.is_some_and(|size| written != size) {
            bail!("incomplete download for {path}");
        }
        Ok((
            format!("{:x}", sha256.finalize()),
            format!("{:x}", git.finalize()),
        ))
    }

    fn publish(
        mut self,
        cache_root: &Path,
        sha256: String,
        git: String,
    ) -> anyhow::Result<(PathBuf, String, String)> {
        let dir = cache_root.join("blobs").join("sha256");
        fs::create_dir_all(&dir)?;
        let blob = dir.join(&sha256);
        if blob.is_file() {
            fs::remove_file(&self.staging.path)?;
        } else {
            // Windows can report a rename failure when a concurrent transfer
            // has already published the same content-addressed blob.
            match fs::rename(&self.staging.path, &blob) {
                Ok(()) => {}
                Err(_) if blob.is_file() => fs::remove_file(&self.staging.path)?,
                Err(error) => return Err(error).context("publishing verified download"),
            }
        }
        self.staging.keep = true;
        Ok((blob, sha256, git))
    }
}

fn has_checksum(expected: Option<&RemoteFile>) -> bool {
    expected.is_some_and(|file| file.sha256.is_some() || file.git_blob_id.is_some())
}

fn strong_etag(response: &Response) -> Option<&str> {
    let tag = response.headers().get(header::ETAG)?.to_str().ok()?;
    let opaque = tag.strip_prefix('"')?.strip_suffix('"')?;
    opaque
        .bytes()
        .all(|byte| byte == b'!' || (b'#'..=b'~').contains(&byte))
        .then_some(tag)
}

fn is_identity(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_ENCODING)
        .is_none_or(|encoding| encoding.as_bytes().eq_ignore_ascii_case(b"identity"))
}

fn remaining_size(response: &Response, offset: u64, partial: &Partial) -> Option<u64> {
    if response.status() != StatusCode::PARTIAL_CONTENT || !is_identity(response) {
        return None;
    }
    if let Some(etag) = &partial.etag
        && response.headers().contains_key(header::ETAG)
        && strong_etag(response) != Some(etag)
    {
        return None;
    }

    let range = response
        .headers()
        .get(header::CONTENT_RANGE)?
        .to_str()
        .ok()?;
    let (span, total) = range.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    let total: u64 = total.parse().ok()?;
    if start.parse::<u64>().ok()? != offset
        || offset >= total
        || end.parse::<u64>().ok()? != total - 1
        || partial.size.is_some_and(|size| size != total)
        || response
            .content_length()
            .is_some_and(|length| length != total - offset)
    {
        return None;
    }
    Some(total)
}
