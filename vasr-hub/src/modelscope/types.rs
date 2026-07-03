use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelScopeResponse {
    #[serde(rename = "Code")]
    #[allow(dead_code)]
    pub code: i64,
    #[serde(rename = "Success")]
    pub success: bool,
    #[serde(rename = "Message")]
    pub message: String,
    #[serde(rename = "Data")]
    pub data: Option<ModelScopeResponseData>,
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
    pub size: u64,
    #[serde(rename = "Sha256")]
    #[allow(dead_code)]
    pub sha256: String,
    #[serde(rename = "Type")]
    pub file_type: String,
}
