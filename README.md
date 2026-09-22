# modelhub

从 ModelScope 和 Hugging Face 下载模型，并列出、清理本地缓存。同一模型只下载一份，再链接到各后端缓存。

## 安装

```bash
cargo install modelhub
```

## 使用

下载模型。默认 revision 在 ModelScope 上是 `master`，在 Hugging Face 上是 `main`。

```bash
modelhub download org/name
modelhub download org/name --revision v1.0.0
modelhub download org/name -j 8
```

权重不一致时，默认只保留一份。`--all-backends` 会两边都留。

列出 modelhub、ModelScope、Hugging Face 缓存里的模型。`$HOME` 显示为 `~`。

```bash
modelhub list
```

清理缓存。不带 `--all` 或 `--backend` 时，只删 modelhub 自己的副本。数据集不会删除。

```bash
modelhub clear org/name
modelhub clear org/name --all
modelhub clear org/name --backend huggingface
modelhub clear --all
modelhub clear --all --backend modelscope
```

`--backend` 可以是 `modelhub`、`modelscope` 或 `huggingface`。`clear --all` 会删除各后端的全部模型目录。
