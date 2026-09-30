use clap::{Parser, Subcommand};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, CellAlignment, ContentArrangement, Table};
use modelhub::ops::RepoEntry;
use modelhub::repos::{CacheSource, RepoStatus};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "modelhub",
    version,
    about = "Download models and datasets from supported hubs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Detect the repository kind, download it once, and link all caches.
    Download {
        /// Model or dataset identifier, for example `org/name`. The kind is detected automatically.
        repo_id: String,
        /// Optional single file to download, for example `README.md` or `data/train.parquet`.
        /// Without it the whole repository is downloaded.
        file: Option<String>,
        /// Revision to request. Defaults to `master` on `ModelScope` and `main` on Hugging Face.
        #[arg(short, long)]
        revision: Option<String>,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Maximum number of files downloaded concurrently.
        #[arg(short = 'j', long, default_value_t = 4, value_parser = parse_jobs)]
        jobs: usize,
        /// Keep both backend versions even when model weights differ.
        #[arg(long)]
        all_backends: bool,
    },
    /// List models and datasets cached by modelhub and by every supported backend.
    List {
        /// Modelhub cache to scan (defaults to `$HOME/.cache/modelhub`).
        /// `ModelScope` and Hugging Face caches are scanned as well.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Verify each modelhub download against its recorded manifest and show a
        /// `STATUS` column.
        #[arg(long)]
        check: bool,
    },
    /// Verify downloads are complete. Fetches and records a manifest for
    /// repositories modelhub did not download itself.
    Check {
        /// Repository to check, for example `org/name`. Defaults to every cached repository.
        repo_id: Option<String>,
        /// Modelhub cache to scan (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Only use recorded manifests; never fetch manifests from the network.
        #[arg(long)]
        offline: bool,
    },
    /// Remove one model or dataset from the modelhub cache, or everything with `--all`.
    Clear {
        /// Model or dataset identifier to remove, for example `org/name`.
        /// Without `--all` or `--backend`, only the modelhub copy is removed.
        #[arg(required_unless_present = "all")]
        model_id: Option<String>,
        /// Remove from every supported backend. With an id, only that id is removed.
        /// Without an id, every model and dataset directory is removed.
        #[arg(long)]
        all: bool,
        /// Only touch this backend: `modelhub`, `modelscope`, or `huggingface`.
        #[arg(long, value_parser = parse_backend)]
        backend: Option<CacheSource>,
        /// Root directory owned by modelhub (defaults to `$HOME/.cache/modelhub`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

/// Parse `--backend` as one supported cache.
fn parse_backend(value: &str) -> Result<CacheSource, String> {
    match value {
        "huggingface" => Ok(CacheSource::HuggingFace),
        "modelhub" => Ok(CacheSource::ModelHub),
        "modelscope" => Ok(CacheSource::ModelScope),
        _ => Err("backend must be modelhub, modelscope, or huggingface".to_owned()),
    }
}

fn parse_jobs(value: &str) -> Result<usize, String> {
    let jobs = value
        .parse::<usize>()
        .map_err(|_| "jobs must be a positive integer".to_owned())?;
    if jobs == 0 {
        return Err("jobs must be at least 1".to_owned());
    }
    Ok(jobs)
}

/// One formatted list row.
struct ListRow {
    id: String,
    kind: &'static str,
    size: String,
    status: Option<String>,
    sources: String,
    paths: Vec<PathBuf>,
}

/// Build display rows from the operations API result.
fn list_rows(entries: &[RepoEntry], check: bool) -> Vec<ListRow> {
    entries
        .iter()
        .map(|entry| ListRow {
            id: entry.id.clone(),
            kind: entry.kind.label(),
            size: human_size(entry.size),
            status: check.then(|| entry.status.unwrap_or(RepoStatus::Unknown).to_string()),
            sources: entry
                .sources
                .iter()
                .map(|source| source.label())
                .collect::<Vec<_>>()
                .join(", "),
            paths: entry.paths.clone(),
        })
        .collect()
}

/// Render the repository list as a bordered table.
fn render_list(rows: &[ListRow], check: bool) {
    println!("{}", render_model_table(rows, None, check));
}

/// Build the list table.
///
/// `width` overrides the terminal width so tests stay stable. `None` lets
/// `comfy-table` measure the terminal and wrap long paths instead of stretching
/// the table past the screen. Several cache paths for one repository stay in the
/// same `PATH` cell, one path per line. `SIZE` is right-aligned. When `check` is
/// set, a `STATUS` column is inserted after `SIZE`.
fn render_model_table(rows: &[ListRow], width: Option<u16>, check: bool) -> String {
    let mut table = Table::new();
    let mut header = vec!["REPO", "KIND", "SIZE"];
    if check {
        header.push("STATUS");
    }
    header.extend(["BACKENDS", "PATH"]);
    table
        .load_style(UTF8_FULL.with_rounded_corners().with_solid_inner_borders())
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(header);
    if let Some(width) = width {
        table.set_width(width);
    }
    // Column 2 is SIZE. Alignment applies to the header and every data cell.
    if let Some(column) = table.column_mut(2) {
        column.set_cell_alignment(CellAlignment::Right);
    }
    for row in rows {
        let path = row
            .paths
            .iter()
            .map(|path| display_path(path))
            .collect::<Vec<_>>()
            .join("\n");
        let mut cells = vec![
            Cell::new(row.id.as_str()),
            Cell::new(row.kind),
            Cell::new(row.size.as_str()),
        ];
        if check {
            cells.push(Cell::new(row.status.as_deref().unwrap_or("unknown")));
        }
        cells.push(Cell::new(row.sources.as_str()));
        cells.push(Cell::new(path));
        table.add_row(cells);
    }
    table.to_string()
}

/// Show `$HOME` as `~` so repeated cache prefixes do not dominate the table.
fn display_path(path: &Path) -> String {
    shorten_home(
        &path.display().to_string(),
        std::env::var("HOME").ok().as_deref(),
    )
}

/// Replace a `home` prefix with `~`.
fn shorten_home(path: &str, home: Option<&str>) -> String {
    if let Some(home) = home.filter(|home| !home.is_empty())
        && let Some(rest) = path.strip_prefix(&format!("{home}/"))
    {
        return format!("~/{rest}");
    }
    path.to_owned()
}

fn human_size(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit = 0;
    let mut divisor = 1u64;
    while size / divisor >= 1024 && unit < UNITS.len() - 1 {
        divisor *= 1024;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        let whole = size / divisor;
        let decimal = (size % divisor) * 10 / divisor;
        format!("{whole}.{decimal} {}", UNITS[unit])
    }
}

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_target(false)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Download {
            repo_id,
            file,
            revision,
            cache_dir,
            jobs,
            all_backends,
        } => {
            let mut options = modelhub::DownloadOptions::new(repo_id);
            options.file = file;
            options.revision = revision;
            if let Some(cache_dir) = cache_dir {
                options.cache_root = cache_dir;
            }
            options.jobs = jobs;
            options.all_backends = all_backends;
            options.progress = true;
            let downloaded = runtime()?.block_on(modelhub::download(&options))?;
            if options.file.is_none() {
                eprintln!("Verified backend manifests and linked compatible cache snapshots.");
            } else {
                let path = downloaded
                    .file
                    .as_deref()
                    .unwrap_or(downloaded.repo_root.as_path());
                println!("Downloaded {} {}", downloaded.kind.label(), path.display());
            }
            Ok(())
        }
        Command::List { cache_dir, check } => {
            let mut options = modelhub::ListOptions {
                check,
                ..modelhub::ListOptions::default()
            };
            if let Some(cache_dir) = cache_dir {
                options.cache_root = cache_dir;
            }
            let entries = modelhub::list(&options)?;
            if entries.is_empty() {
                println!(
                    "No cached models or datasets in modelhub ({}), ModelScope ({}), or Hugging Face ({})",
                    options.cache_root.display(),
                    modelhub::modelscope::cache_dir().display(),
                    modelhub::huggingface::cache_dir().display()
                );
                return Ok(());
            }
            render_list(&list_rows(&entries, check), check);
            Ok(())
        }
        Command::Check {
            repo_id,
            cache_dir,
            offline,
        } => {
            let mut options = modelhub::CheckOptions {
                repo_id,
                offline,
                ..modelhub::CheckOptions::default()
            };
            if let Some(cache_dir) = cache_dir {
                options.cache_root = cache_dir;
            }
            let entries = runtime()?.block_on(modelhub::check(&options))?;
            if entries.is_empty() {
                println!("No matching cached repositories");
                return Ok(());
            }
            render_list(&list_rows(&entries, true), true);
            Ok(())
        }
        Command::Clear {
            model_id,
            all,
            backend,
            cache_dir,
        } => {
            let mut options = modelhub::ClearOptions {
                repo_id: model_id.clone(),
                all,
                backend,
                ..modelhub::ClearOptions::default()
            };
            if let Some(cache_dir) = cache_dir {
                options.cache_root = cache_dir;
            }
            let summary = modelhub::clear(&options)?;
            for path in &summary.removed {
                eprintln!("Removed {}", path.display());
            }
            let labels = summary
                .targets
                .iter()
                .map(|source| source.label())
                .collect::<Vec<_>>()
                .join(", ");
            if !summary.found {
                let model_id = model_id.unwrap_or_default();
                println!("`{model_id}` is not cached in {labels}");
            } else if model_id.is_none() {
                println!("Cleared models and datasets from {labels}");
            } else {
                let model_id = model_id.unwrap_or_default();
                println!("Cleared `{model_id}` from {labels}");
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, ListRow, human_size, render_model_table, shorten_home};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn formats_cache_sizes() {
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(1536), "1.5 KiB");
    }

    #[test]
    fn renders_list_as_a_bordered_table() {
        let rows = vec![
            ListRow {
                id: "acme/demo".to_owned(),
                kind: "model",
                size: "1.5 KiB".to_owned(),
                status: None,
                sources: "huggingface".to_owned(),
                paths: vec![PathBuf::from("/cache/hf")],
            },
            ListRow {
                id: "org/name".to_owned(),
                kind: "dataset",
                size: "2 B".to_owned(),
                status: None,
                sources: "modelscope".to_owned(),
                paths: vec![PathBuf::from("/cache/ms"), PathBuf::from("/cache/ms2")],
            },
        ];
        // A fixed width keeps the snapshot independent of the terminal size.
        let table = render_model_table(&rows, Some(64), false);
        assert_eq!(
            table,
            "\
╭───────────┬─────────┬─────────┬─────────────┬────────────╮
│ REPO      │ KIND    │    SIZE │ BACKENDS    │ PATH       │
╞═══════════╪═════════╪═════════╪═════════════╪════════════╡
│ acme/demo │ model   │ 1.5 KiB │ huggingface │ /cache/hf  │
├───────────┼─────────┼─────────┼─────────────┼────────────┤
│ org/name  │ dataset │     2 B │ modelscope  │ /cache/ms  │
│           │         │         │             │ /cache/ms2 │
╰───────────┴─────────┴─────────┴─────────────┴────────────╯"
        );
        assert_eq!(
            shorten_home("/Users/me/.cache/huggingface/hub", Some("/Users/me")),
            "~/.cache/huggingface/hub"
        );
    }

    #[test]
    fn renders_status_column_when_checking() {
        let rows = vec![
            ListRow {
                id: "acme/demo".to_owned(),
                kind: "model",
                size: "1.5 KiB".to_owned(),
                status: Some("complete".to_owned()),
                sources: "huggingface".to_owned(),
                paths: vec![PathBuf::from("/cache/hf")],
            },
            ListRow {
                id: "org/name".to_owned(),
                kind: "dataset",
                size: "2 B".to_owned(),
                status: Some("incomplete 1/2".to_owned()),
                sources: "modelscope".to_owned(),
                paths: vec![PathBuf::from("/cache/ms")],
            },
        ];
        let checked = render_model_table(&rows, Some(120), true);
        assert!(checked.contains("STATUS"));
        assert!(checked.contains("complete"));
        assert!(checked.contains("incomplete 1/2"));
        // Without `--check` the column disappears entirely.
        assert!(!render_model_table(&rows, Some(120), false).contains("STATUS"));
    }

    #[test]
    fn clear_requires_a_model_or_all() {
        assert!(Cli::try_parse_from(["modelhub", "clear"]).is_err());
        assert!(Cli::try_parse_from(["modelhub", "clear", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "clear", "acme/demo", "--all"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "clear", "--all"]).is_ok());
        assert!(
            Cli::try_parse_from(["modelhub", "clear", "acme/demo", "--backend", "huggingface"])
                .is_ok()
        );
        assert!(
            Cli::try_parse_from(["modelhub", "clear", "--all", "--backend", "modelscope"]).is_ok()
        );
        assert!(Cli::try_parse_from(["modelhub", "clear", "--backend", "huggingface"]).is_err());
        assert!(Cli::try_parse_from(["modelhub", "clear", "--backend", "nope"]).is_err());
    }

    #[test]
    fn download_jobs_must_be_positive() {
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "8"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "--jobs", "0"]).is_err());
    }

    #[test]
    fn download_accepts_an_optional_file() {
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "download", "acme/demo", "README.md"]).is_ok());
        assert!(
            Cli::try_parse_from(["modelhub", "download", "acme/demo", "data/train.parquet"])
                .is_ok()
        );
    }

    #[test]
    fn list_accepts_a_check_flag() {
        assert!(Cli::try_parse_from(["modelhub", "list"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "list", "--check"]).is_ok());
    }

    #[test]
    fn check_accepts_a_repository_and_offline() {
        assert!(Cli::try_parse_from(["modelhub", "check"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "check", "acme/demo"]).is_ok());
        assert!(Cli::try_parse_from(["modelhub", "check", "--offline"]).is_ok());
    }
}
