# modelhub

从 ModelScope 和 Hugging Face 下载模型和数据集，并列出、清理本地缓存。同一仓库只下载一份，再链接到各后端缓存。

## 安装

```bash
cargo install modelhub
```

## 使用

下载仓库。`modelhub` 会自动识别 `org/name` 是模型还是数据集。默认 revision 在 ModelScope 上是 `master`，在 Hugging Face 上是 `main`。

```bash
modelhub download org/name
modelhub download org/name --revision v1.0.0
modelhub download org/name -j 8
```

只下载仓库里的单个子文件（路径相对仓库根目录，可含子目录）：

```bash
modelhub download org/name README.md
modelhub download org/name data/train.parquet
```

单文件下载只写入 modelhub 缓存并打印落盘路径，不会链接到 ModelScope / Hugging Face 原生缓存（避免后端把不完整的快照当成整库）。

权重不一致时，默认只保留一份。`--all-backends` 会两边都留。

列出 modelhub、ModelScope、Hugging Face 缓存里的模型和数据集（`KIND` 列区分）。`$HOME` 显示为 `~`。

```bash
modelhub list
```

`list --check` 会离线核对仓库是否完整并新增 `STATUS` 列：下载成功时记录了 `.modelhub-manifest.json`，按它逐一检查文件是否存在、大小是否一致。

```bash
modelhub list --check
```

专门的 `check` 子命令除了核对，还会**为没有清单的仓库生成清单**——即在 modelhub 之外下载的原生缓存（HF / ModelScope 自己下的）。它联网拉取远端文件清单，写入该仓库目录下的 `.modelhub-manifest.json`，之后再核对。可以只检查一个仓库，或用 `--offline` 跳过联网。

```bash
modelhub check
modelhub check org/name
modelhub check --offline
```

`STATUS` 取值：

- `complete`：清单里的文件全部齐全且大小一致。
- `incomplete 34/36`：还差若干文件。
- `unknown`：没有清单（如跳过联网、拉取失败，或单文件下载）。

> 生成的清单会写到原生缓存目录里（例如 `~/.cache/huggingface/hub/models--org--name/.modelhub-manifest.json`），`clear` 删除该目录时会一并移除。


清理缓存。不带 `--all` 或 `--backend` 时，只删 modelhub 自己的副本。同一个 `org/name` 若是模型和数据集都会一并匹配。

```bash
modelhub clear org/name
modelhub clear org/name --all
modelhub clear org/name --backend huggingface
modelhub clear --all
modelhub clear --all --backend modelscope
```

`--backend` 可以是 `modelhub`、`modelscope` 或 `huggingface`。`clear --all` 会删除各后端的全部模型和数据集目录。

## 上传

把本地文件或目录上传到模型 / 数据集仓库。上传时必须显式指定 `--repo-type`;默认上传到所有已配置凭据的后端(HF 用 `HF_TOKEN`,或 `~/.cache/huggingface/token`;ModelScope 用 `MODELSCOPE_API_TOKEN`,或 `modelscope login` 写入的 `~/.modelscope/credentials/cookies`)。仓库不存在时默认自动创建,`--no-create` 可关闭。

> 注意:`~/.modelscope/credentials/session` 是 SDK 的匿名安装标识,**不是** token,modelhub 不会读取它。

```bash
modelhub upload org/name ./output --repo-type model
modelhub upload org/name ./weights.safetensors --repo-type model --path-in-repo weights.safetensors
modelhub upload org/name ./data --repo-type dataset --backend modelscope --revision master
modelhub upload org/name ./output --repo-type model --private --include '*.safetensors'
```

- 普通小文件内联进提交;大文件(或 LFS 后缀)走 LFS:HF 支持 basic 与 multipart 分块,ModelScope 走预签名 OSS。
- 多文件会按内联体积与服务端上限自动拆成多次提交(ModelScope 单次提交上限 2000 个操作)。
- `--include/--exclude` 用 glob 过滤;`--backend` 可重复以指定后端。
- **默认增量更新**:每次先拉目标 revision 的远端清单做 diff,只传新增/修改的文件,内容未变化的跳过;没有任何变化时**不产生提交**。
  - `--delete`:把远端有、本地没有的文件删掉。删除限定作用域——有 `--path-in-repo` 就只删该子树,否则要求输入是目录(避免单文件上传误删整库)。
  - `--dry-run`:只打印 `+added ~modified =unchanged -deleted`,不上传、不建仓、不提交。
  - `--force`:忽略 diff,全量覆盖(旧行为)。
- `--revision/-r <branch>` 指定目标分支/版本(默认 HF `main`、ModelScope `master`)。HF 上分支不存在会**自动从 `main` 创建**;ModelScope 的 revision 是从 master 派生的 tag,建议先传 `master` 再打 tag。
- HF 的 Xet 存储模式暂不支持,遇到会明确报错并提示改用 `huggingface-cli`。

## 库 API

四个子命令都有对应的库函数,`download` / `check` 是 async,`list` / `clear` 是同步;它们返回结构化数据,不直接打印。

```rust
use modelhub::{CheckOptions, ClearOptions, DownloadOptions, ListOptions, RepoStatus};

// 列出缓存(可选校验完整性)
let options = ListOptions { check: true, ..Default::default() };
for entry in modelhub::list(&options)? {
    let status = entry.status.unwrap_or(RepoStatus::Unknown);
    println!("{}\t{}\t{status}", entry.id, entry.kind.label());
}

// 下载(需 async 运行时)
let mut options = DownloadOptions::new("org/name");
options.file = Some("README.md".to_owned());
let downloaded = runtime.block_on(modelhub::download(&options))?;

// 校验并为原生缓存生成清单
let checked = runtime.block_on(modelhub::check(&CheckOptions::default()))?;

// 清理并拿到删除了什么
let summary = modelhub::clear(&ClearOptions { repo_id: Some("org/name".into()), ..Default::default() })?;
```

- `list` / `check` 返回 `Vec<RepoEntry>`,含 `kind`、`size`、`status`、`sources`、`paths`、`hits`。
- `download` 返回 `DownloadedRepo`(仓库根目录、各后端快照根、单文件路径)。
- `clear` 返回 `ClearSummary { targets, removed, found }`。
- `upload` 返回 `UploadSummary { results }`,每个后端一条 `BackendUpload`(`created`、`revision`、`commit`、`uploaded`、`skipped`、`bytes`、`counts`);`UploadOptions` 的 `delete` / `dry_run` / `force` 控制同步行为。
- 需要显式指定缓存的场景,可用 `*Options` 里的 `cache_root`、`modelscope_cache`、`huggingface_hub` 覆盖默认值。

```rust
// 上传(需 async 运行时)
let options = modelhub::UploadOptions::new(
    "org/name",
    modelhub::RepoKind::Model,
    vec!["./output".into()],
);
let summary = runtime.block_on(modelhub::upload(&options))?;
```


