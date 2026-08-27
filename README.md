# modelhub

`modelhub` 是一个 Rust library crate，用来在 Rust 项目中下载并缓存模型仓库资源。

当前已支持 `ModelScope` 和 `Hugging Face`（支持 `HF_ENDPOINT`/`hf-mirror`）：

- 模型仓库下载
- 数据集仓库下载
- 模型/数据集单文件下载
- revision 指定
- 与 `ModelScope` SDK/CLI 一致的缓存目录结构

上传和更多后端会在后续版本中继续补齐。

## 安装

```toml
[dependencies]
modelhub = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

安装 CLI（`cargo build` 只在项目的 `target/` 下编译，不会复制到 PATH）

```bash
cargo install --path .
```

安装后也会提供 `modelhub` 命令行工具。它会自动探测 ModelScope 和 Hugging Face，下载一次并写入统一的内容寻址缓存，再为各后端生成自己的快照目录。两个后端都有模型时，会逐文件比较大小、SHA-256；对于没有 SHA-256 元数据的普通 Git 文件，还会比较 Hugging Face Git blob 哈希。确认一致的文件只保存一份，两个后端共同复用。

```bash
modelhub download org/model-name
modelhub download org/model-name --revision v1.0.0
modelhub download org/model-name --cache-dir /data/modelhub-cache
modelhub download org/model-name --jobs 8
# 后端关键文件不一致时，保留两边版本而不是终止
modelhub download org/model-name --all-backends
```

模型文件默认最多同时下载 4 个，`--jobs` 可调整并发数。若两个后端的权重、配置或 tokenizer 等关键文件哈希不一致，命令会停止并提示；确认需要保留两套版本时可使用 `--all-backends`。

查看与清理缓存：

```bash
modelhub list
modelhub clear org/model-name
modelhub clear --all
```

`list` 会显示模型 ID、占用空间和缓存路径。`clear <model-id>` 只删除指定模型，
`clear --all` 删除整个 modelhub 缓存。清理时会移除由 modelhub 创建、并指向统一缓存的
后端符号链接；已存在的独立 ModelScope/Hugging Face 真实缓存目录会被保留。

命令会把下载内容统一保存到 `$HOME/.cache/modelhub`（可用 `MODELHUB_CACHE` 或
`--cache-dir` 修改）。文件实体位于 `blobs/sha256/<digest>`，后端快照位于
`models/<namespace--name>/{modelscope,huggingface}/snapshots/<revision>`，再将兼容的
ModelScope 和 Hugging Face 原生缓存目录链接到对应快照。模型只存在于一个后端时，只创建该后端的快照，不会伪造另一后端的 commit。
如果目标后端缓存目录已经存在，命令会显示提醒并保留该目录，再补齐缺失的快照文件，不会因目录存在而报错；统一缓存中已有的内容也会直接复用。
需要 Hugging Face 私有模型时设置 `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN`；ModelScope
登录状态沿用 `$HOME/.modelscope/config/cookies`。

Hugging Face 请求默认先访问 `hf-mirror.com`，镜像失败时自动回退到官方站点；也可以通过
`HF_ENDPOINT` 指定单一地址：

```bash
HF_ENDPOINT=https://hf-mirror.com modelhub download org/model-name
```

当前 `download` 命令支持模型仓库自动探测；数据集和更多后端仍可通过 Rust API 使用，后续会继续补齐。

## 快速开始

```rust
use modelhub::modelscope;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    modelscope::download_model(
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        modelscope::cache_dir(),
    )
    .await?;

    Ok(())
}
```

默认会下载到：

```text
$HOME/.cache/modelscope/models/iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master
```

## 缓存目录

`modelscope::cache_dir()` 返回 `ModelScope` 后端的缓存根目录。

优先级：

1. 当前进程中通过 `modelscope::set_cache_dir(...)` 设置的目录
2. 环境变量 `MODELSCOPE_CACHE`
3. `$HOME/.cache/modelscope`
4. 当 `$HOME` 不存在时，使用 `/tmp/.cache/modelscope`

缓存布局与官方 `ModelScope` SDK/CLI 保持一致：

```text
<cache>/models/<namespace--model>/snapshots/<revision>
<cache>/datasets/<namespace--dataset>/snapshots/<revision>
```

示例：

```rust
use modelhub::modelscope;

modelscope::set_cache_dir("/data/modelscope-cache");
assert_eq!(
    modelscope::cache_dir(),
    std::path::PathBuf::from("/data/modelscope-cache")
);
```

## 下载模型

下载默认 `master` revision：

```rust
modelhub::modelscope::download_model(
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

下载指定 revision：

```rust
modelhub::modelscope::download_model_revision(
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "v1.0.0",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

## 下载数据集

下载默认 `master` revision：

```rust
modelhub::modelscope::download_dataset(
    "modelscope/clue",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

下载指定 revision：

```rust
modelhub::modelscope::download_dataset_revision(
    "modelscope/clue",
    "master",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

## 下载单文件

单文件下载会返回本地文件路径。

模型文件：

```rust
let path = modelhub::modelscope::download_model_file(
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "configuration.json",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

指定 revision 的模型文件：

```rust
let path = modelhub::modelscope::download_model_file_revision(
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "configuration.json",
    "v1.0.0",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

数据集文件：

```rust
let path = modelhub::modelscope::download_dataset_file(
    "modelscope/clue",
    "README.md",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

指定 revision 的数据集文件：

```rust
let path = modelhub::modelscope::download_dataset_file_revision(
    "modelscope/clue",
    "README.md",
    "master",
    modelhub::modelscope::cache_dir(),
)
.await?;
```

## API

### 缓存

```rust
modelscope::cache_dir() -> PathBuf
modelscope::set_cache_dir(dir: impl Into<PathBuf>)
```

### 整仓下载

```rust
modelscope::download_model(model_id, save_dir).await
modelscope::download_model_revision(model_id, revision, save_dir).await
modelscope::download_model_revision_with_concurrency(model_id, revision, save_dir, concurrency).await

modelscope::download_dataset(dataset_id, save_dir).await
modelscope::download_dataset_revision(dataset_id, revision, save_dir).await
```

### 单文件下载

```rust
modelscope::download_model_file(model_id, file_path, save_dir).await
modelscope::download_model_file_revision(model_id, file_path, revision, save_dir).await

modelscope::download_dataset_file(dataset_id, file_path, save_dir).await
modelscope::download_dataset_file_revision(dataset_id, file_path, revision, save_dir).await
```

### Hugging Face

```rust
modelhub::huggingface::download_model(model_id, revision, modelhub_cache).await
modelhub::huggingface::download_model_with_concurrency(model_id, revision, modelhub_cache, concurrency).await
modelhub::huggingface::cache_dir() -> std::path::PathBuf
```

`revision` 为 `Option<&str>`，为空时使用 Hugging Face 的 `main` 分支。

## 特点

- 保持 `ModelScope` 官方缓存布局
- 支持模型仓库和数据集仓库
- 支持整仓下载和单文件下载
- 支持 revision
- 自动创建本地缓存目录
- 保留远端仓库中的子目录结构
- 已完整缓存的文件会跳过，不重复下载
- 多文件并发下载
- 下载时显示进度条
- 支持读取 `$HOME/.modelscope/config/cookies`，用于需要登录权限的资源
- 使用临时 `.part` 文件下载，成功后再替换最终文件，避免失败下载污染缓存

## 是否支持断点重传？

目前不支持真正的 byte-range 断点续传。

当前行为：

- 如果目标文件已存在且大小正确，会直接复用
- 如果目标文件不存在，会重新下载
- 如果存在上次中断留下的 `.part` 文件，会删除后重新下载
- 下载完成后会检查字节数是否与远端返回的文件大小一致

所以当前支持的是完整文件缓存复用和安全重试，不是从上次中断位置继续下载。

## 兼容导出

为了兼容旧代码，仍然保留了这些顶层导出：

```rust
use modelhub::cache_dir;
use modelhub::download;
use modelhub::modelscope_cache_dir;
use modelhub::set_cache_dir;
```

新代码建议使用后端命名空间：

```rust
use modelhub::modelscope;
```
