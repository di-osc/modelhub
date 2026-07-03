# modelhub

`modelhub` 是一个 Rust library crate，用来在 Rust 项目中下载并缓存模型仓库资源。

当前已支持 `ModelScope`：

- 模型仓库下载
- 数据集仓库下载
- 模型/数据集单文件下载
- revision 指定
- 与 `ModelScope` SDK/CLI 一致的缓存目录结构

上传和更多后端，例如 `hf-mirror`，会在后续版本中继续补齐。

## 安装

```toml
[dependencies]
modelhub = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

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
