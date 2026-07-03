# modelhub

Rust helpers for downloading and caching models from model hubs. The current
crate is `modelhub`, with `ModelScope` download support.

## Usage

```rust
use modelhub::{cache_dir, modelscope};

# async fn run() -> anyhow::Result<()> {
modelscope::download_model("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", cache_dir()).await?;
# Ok(())
# }
```

Models are stored under `MODELHUB_CACHE_DIR` when it is set. For compatibility,
`VASR_MODEL_DIR` is also respected. Otherwise the default cache root is
`$HOME/.cache/modelhub`.

## Release

The GitHub workflow in `.github/workflows/publish-crate.yml` publishes
`modelhub` to crates.io.

1. Add the `CARGO_REGISTRY_TOKEN` repository secret.
2. Create the `crates-io` GitHub environment if environment approval is needed.
3. Create and publish a GitHub release after updating the workspace version.

The workflow can also be run manually with `dry_run` enabled to validate the
package without publishing.
