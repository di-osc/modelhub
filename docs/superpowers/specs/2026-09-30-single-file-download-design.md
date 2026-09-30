# 单文件下载快速路径设计（issue #1）

日期：2026-09-30
状态：已与维护者对齐

## 背景与问题

`DownloadOptions.file` 声称"只下载一个文件"，但 `download()` 在请求这个文件之前总会先
`download_repo` → `detect_manifests`：四路探测（Hugging Face × ModelScope × model ×
dataset），ModelScope 数据集还要按每页 200 条翻完整个 `repo/tree`。缓存里已有该文件时
同样要走完这些网络请求；单文件下载也不写 `.modelhub-manifest.json`，没有可离线复用的
记录。asr-data 取一个几十 KB 的音频文件也要等完整文件树，且每次调用都重做。

## 目标

- 单文件下载：探测的最小单位是"这一个文件的请求"，绝不调用任何列举接口。
- 缓存命中零网络，直接返回本地路径。
- 调用方可以通过 `kind` + `backend` 完全消除探测；不给时自动检测，但检测也不列库。
- 整仓下载默认行为不变，`kind`/`backend` 可选收窄。
- 不新增公开入口，只修改 `download()` 与 `DownloadOptions`。

## 非目标

- 不做单文件的 `.modelhub-manifest.json` 记录与离线校验。
- 不把单文件链接到 `~/.cache/modelscope` 或 Hugging Face 原生缓存（与现状一致）。
- 不重构 `upload`、`check`、`list`、`clear`。

## API

### `DownloadOptions`（`src/ops.rs`）

```rust
pub struct DownloadOptions {
    pub repo_id: String,
    pub file: Option<String>,       // Some → 单文件路径
    pub kind: Option<RepoKind>,     // None → 该维度自动检测
    pub backend: Option<Backend>,   // None → 该维度自动检测
    pub revision: Option<String>,   // None → HF 用 main，MS 用 master
    pub cache_root: PathBuf,
    pub jobs: usize,                // 仅整仓
    pub all_backends: bool,         // 仅整仓
    pub progress: bool,
}
```

- `DownloadOptions::new(repo_id)` 签名不变，新字段默认 `None`。
- `Backend` 复用 `unified.rs` 现有私有枚举（`HuggingFace | ModelScope`），提升为公开
  类型并从 `lib.rs` 导出为 `modelhub::Backend`。
- `DownloadedRepo` 结构不变。

### CLI（`src/main.rs`）

`modelhub download <repo_id> [file] [--repo-type model|dataset] [--backend huggingface|modelscope]`

两个 flag 均可选，翻译成对应的 `kind`/`backend` 字段。给了 `file` 不强制传 flag，
不传就走自动检测。

## 行为

### 单文件（`file: Some(...)`）

1. **校验（任何请求之前）**：`file` 或 `revision` 含 `..` 组件或为绝对路径 → 立即
   返回错误。复用/扩展 `safe_path` 的校验逻辑。
2. **缓存命中**：查候选快照路径
   `{cache}/{models|datasets}/{repo--id}/{huggingface|modelscope}/snapshots/{revision}/{file}`。
   `kind`/`backend` 已知则只查那一条；未知则查该维度全部候选。`revision` 缺省时按
   后端各自取 `main`（HF）/`master`（MS）。任一命中 → 直接 `Ok`，零网络。
   快照目录使用请求的 revision 字符串（不做 commit sha 解析），否则离线命中做不到。
3. **未命中**：按已知维度收窄，并行对候选组合发单文件 GET（组合数 =
   `kind` 已知 ? 1 : 2 乘 `backend` 已知 ? 1 : 2，最多 4）。先成功者赢，其余立即取消。
   全部失败 → 报错并列出尝试过的 URL。
   - ModelScope：`https://modelscope.cn/api/v1/{datasets|models}/{repo_id}/repo?Revision=...&FilePath=...`
   - Hugging Face：`{endpoint}/{models|datasets}/{repo_id}/resolve/{revision}/{file}`
4. **落盘**：走现有 `materialize` 管线：staging → `blobs/sha256/<hash>` →
   hard-link 到快照路径。
5. 不写 `.modelhub-manifest.json`，不链接任何原生缓存。

注：并行候选下若多个组合存在同名文件，取先成功者赢；不复刻整仓路径"两类同时存在则
报错"的语义，文档中说明。

返回值：`DownloadedRepo.kind` 为实际命中的类型，`file: Some(本地路径)`；
`huggingface_root`/`modelscope_root` 只填实际命中的后端快照根；`repo_root` 照旧。

### 整仓（`file: None`）

| `kind` | `backend` | 行为 |
|---|---|---|
| `None` | `None` | 现状：4 路探测、跨后端校验、写 manifest、链原生缓存。 |
| `Some(k)` | `None` | 只探类型 `k` 在两个后端的清单；校验/manifest/链接不变。 |
| `None` | `Some(b)` | 只探后端 `b` 上的 model/dataset 两种；其余不变。 |
| `Some(k)` | `Some(b)` | 零探测，只拉该后端该类型的清单和文件；无跨后端校验，`all_backends` 无效。 |

两类型同时存在沿用现有报错（`exists as both ...`）；找不到时报
`was not found ...`。`jobs == 0` 依旧报错。

## 实现要点

- `unified.rs`：`download_repo` 移除 `file` 参数；单文件逻辑独立为内部函数
  （校验 + 缓存命中 + 并行单文件请求 + 落盘）。`detect_manifests` 增加按
  `kind`/`backend` 收窄的能力（或拆出窄版本）。
- `Backend` 提升公开；`ops.rs` 的 `download` 按 `file` 分派两条路径，整仓路径继续做
  原生缓存链接。
- ModelScope base URL 允许 `MODELSCOPE_ENDPOINT` 环境变量覆盖（与现有 `HF_ENDPOINT`
  对称），用于测试注入；默认 `https://modelscope.cn`。
- 进度条：单文件路径同样受 `progress` 控制。

## 错误面（均在副作用前可预期）

- 路径穿越/绝对路径 → 校验错误（网络请求之前）。
- 候选全部失败 → "not found"错误，含尝试过的 URL。
- 下载完整性失败（大小不符）→ 沿用 `materialize` 的错误。

## 测试

- 单元（`src/` 内）：路径校验拒绝 `..`/绝对路径；单文件 URL 拼装；缓存命中零网络
  （预置快照文件离线断言）。
- 集成（`tests/`，本地 mock server + `HF_ENDPOINT`/`MODELSCOPE_ENDPOINT`）：
  - `kind`/`backend` 收窄后候选请求数正确（1/2/4）；
  - 先成功者赢、其余取消；
  - 单文件不写 `.modelhub-manifest.json`、不链接原生缓存；
  - 整仓矩阵各分支的探测与 manifest 行为保持既有语义。
- 现有 `tests/api.rs` 与 `main.rs` 单元测试随签名更新。
