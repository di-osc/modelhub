# modelhub

`modelhub` 是一个用于从模型仓库下载并缓存模型文件的 Rust crate。

当前版本已实现 `ModelScope` 模型下载。数据集下载、上传、更多 hub 后端还在后续计划中。

## 安装

```toml
[dependencies]
modelhub = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## 快速开始

```rust
use modelhub::{cache_dir, modelscope};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    modelscope::download_model(
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        cache_dir(),
    )
    .await?;

    Ok(())
}
```

上面的模型会被下载到：

```text
$HOME/.cache/modelhub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

## 缓存目录

`cache_dir()` 返回默认缓存根目录。

优先级如下：

1. 通过 `set_cache_dir(...)` 在当前进程中设置的目录
2. 环境变量 `MODELHUB_CACHE_DIR`
3. 环境变量 `VASR_MODEL_DIR`，保留用于兼容旧代码
4. `$HOME/.cache/modelhub`
5. 当 `$HOME` 不存在时，使用 `/tmp/.cache/modelhub`

示例：

```rust
use modelhub::{cache_dir, set_cache_dir};

set_cache_dir("/data/models");
assert_eq!(cache_dir(), std::path::PathBuf::from("/data/models"));
```

## API

### `cache_dir() -> PathBuf`

返回 `modelhub` 使用的缓存根目录。

注意它只是根目录。实际模型文件会放在：

```text
cache_dir/<model_id>
```

### `set_cache_dir(dir: impl Into<PathBuf>)`

设置当前进程的缓存根目录。

适合应用程序自己控制模型目录，而不是依赖环境变量。

### `modelscope::download_model(model_id, save_dir).await`

从 `ModelScope` 下载一个模型仓库中的所有文件。

参数：

- `model_id`：`ModelScope` 模型 ID，例如
  `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `save_dir`：缓存根目录，通常传 `cache_dir()`

下载结果：

```text
save_dir/<model_id>/<repo_file_path>
```

例如：

```rust
modelhub::modelscope::download_model("damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "/data/models").await?;
```

文件会被保存到：

```text
/data/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

## 特点

- 支持 `ModelScope` 模型仓库下载
- 自动创建本地缓存目录
- 保留远端仓库中的子目录结构
- 已完整缓存的文件会跳过，不重复下载
- 多文件并发下载
- 下载时显示进度条
- 支持读取 `$HOME/.modelscope/config/cookies`，用于需要登录权限的模型
- 使用临时 `.part` 文件下载，成功后再替换最终文件，避免失败下载污染缓存

## 是否支持断点重传？

目前不支持真正的 byte-range 断点续传。

当前行为是：

- 如果目标文件已存在且大小正确，会直接复用
- 如果目标文件不存在，会重新下载
- 如果存在上次中断留下的 `.part` 文件，会删除后重新下载
- 下载完成后会检查字节数是否与 `ModelScope` 返回的文件大小一致

所以当前支持的是“完整文件缓存复用”和“安全重试”，不是从上次中断位置继续下载。

## 兼容导出

为了兼容旧代码，仍然保留了这些导出：

```rust
use modelhub::download;
use modelhub::modelscope_cache_dir;
```

新代码建议使用：

```rust
use modelhub::{cache_dir, modelscope};
```
