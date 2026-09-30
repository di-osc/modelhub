use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelScopeResponse {
    #[serde(rename = "Code")]
    #[allow(dead_code)]
    pub code: i64,
    // Dataset tree responses omit `Success`; treat a missing flag as success and
    // rely on the `Data` payload and HTTP status.
    #[serde(rename = "Success", default = "default_true")]
    pub success: bool,
    #[serde(rename = "Message", default)]
    pub message: String,
    #[serde(rename = "Data")]
    pub data: Option<ModelScopeResponseData>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ModelScopeResponseData {
    #[serde(rename = "Files")]
    pub files: Vec<RepoFile>,
}

#[derive(Debug, Deserialize)]
pub struct RepoFile {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Path")]
    pub path: String,
    #[serde(rename = "Size")]
    #[serde(default)]
    pub size: u64,
    #[serde(rename = "Sha256")]
    #[serde(default)]
    #[allow(dead_code)]
    pub sha256: Option<String>,
    #[serde(rename = "Type")]
    #[serde(default)]
    pub file_type: String,
}
