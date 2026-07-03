use anyhow::Context;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const COOKIES_FILE: &str = "cookies";

pub const USER_AGENT: (&str, &str) = (
    "User-Agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
);

pub async fn http_client() -> anyhow::Result<reqwest::Client> {
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));
    let mut default_headers = reqwest::header::HeaderMap::new();

    if let Some(cookies) = read_cookies()? {
        default_headers.insert("Cookie", cookies.parse()?);
    }

    Ok(client.default_headers(default_headers).build()?)
}

fn read_cookies() -> anyhow::Result<Option<String>> {
    let cookies_file = cookies_dir()?.join(COOKIES_FILE);
    if !cookies_file.exists() {
        return Ok(None);
    }

    let cookies = fs::read_to_string(&cookies_file)?;
    let cookies: serde_json::Value = serde_json::from_str(&cookies)?;
    let cookies = cookies
        .as_object()
        .context("failed to parse ModelScope cookies")?
        .iter()
        .map(|(key, value)| format!("{}={}", key, value.as_str().unwrap_or_default()))
        .collect::<Vec<_>>()
        .join("; ");

    Ok(Some(cookies))
}

fn cookies_dir() -> anyhow::Result<PathBuf> {
    let dir = std::env::var("HOME")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join(".modelscope")
        .join("config");

    fs::create_dir_all(&dir)?;
    Ok(dir)
}
