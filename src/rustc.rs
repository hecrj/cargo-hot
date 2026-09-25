use serde::{Deserialize, Serialize};
use std::{
    env::{args, vars},
    path::{Path, PathBuf},
};

/// The environment variable indicating where the rustc args directory is located.
///
/// When `cargo-hot` runs as a rustc wrapper, it writes the arguments of every crate it wraps to
/// this directory as `{crate_name}.{lib|bin}.json` files.
pub const DX_RUSTC_WRAPPER_ENV_VAR: &str = "DX_RUSTC";

/// Is `cargo-hot` being used as a rustc wrapper?
///
/// This is primarily used to intercept cargo, enabling fast hot-patching by caching the
/// environment cargo sets up for every crate in the workspace.
///
/// In a different world we could simply rely on cargo printing link args and the rustc command,
/// but it doesn't seem to output that in a reliable, parseable, cross-platform format (ie using
/// command files on windows...), so we're forced to do this interception nonsense.
pub fn is_wrapping_rustc() -> bool {
    std::env::var(DX_RUSTC_WRAPPER_ENV_VAR).is_ok()
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Args {
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
    /// The working directory the rustc process was invoked in. Thin builds replay rustc
    /// in this directory so relative paths in the captured args resolve the same way.
    #[serde(default)]
    pub cwd: PathBuf,
}

/// The rustc args captured for every crate during a build, plus the linker args of the tip
/// crate's final link invocation.
///
/// The `rustc_args` map is keyed by `{crate_name}.{suffix}` where the suffix is `lib` for
/// lib/rlib crate types and `bin` otherwise, matching the per-crate files the wrapper writes.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceRustcArgs {
    pub link_args: Vec<String>,
    pub rustc_args: std::collections::HashMap<String, Args>,
}

impl WorkspaceRustcArgs {
    pub fn new(link_args: Vec<String>) -> Self {
        Self {
            link_args,
            rustc_args: Default::default(),
        }
    }
}

/// Check if the arguments indicate a linking step, including those in command files.
fn has_linking_args() -> bool {
    for arg in std::env::args() {
        // Direct check for linker-like arguments
        if arg.ends_with(".o") || arg == "-flavor" {
            return true;
        }

        // Check inside command files
        if let Some(path_str) = arg.strip_prefix('@')
            && let Ok(file_binary) = std::fs::read(path_str)
        {
            // Handle both UTF-8 and UTF-16LE encodings for response files.
            let content = String::from_utf8(file_binary.clone()).unwrap_or_else(|_| {
                let binary_u16le: Vec<u16> = file_binary
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|a| u16::from_le_bytes([a[0], a[1]]))
                    .collect();
                String::from_utf16_lossy(&binary_u16le)
            });

            // Check if any line in the command file contains linking indicators.
            if content.lines().any(|line| {
                let trimmed_line = line.trim().trim_matches('"');
                trimmed_line.ends_with(".o") || trimmed_line == "-flavor"
            }) {
                return true;
            }
        }
    }

    false
}

/// Run rustc directly, but output the result to a file.
///
/// <https://doc.rust-lang.org/cargo/reference/config.html#buildrustc>
pub fn run_rustc() {
    // If we are being asked to link, delegate to the linker action.
    if has_linking_args() {
        crate::link::LinkAction::from_env()
            .expect("Linker action not found")
            .run_link();
        return;
    }

    let args_dir: PathBuf = std::env::var(DX_RUSTC_WRAPPER_ENV_VAR)
        .expect("DX_RUSTC not set")
        .into();

    let cwd = std::env::current_dir().unwrap_or_default();

    let rustc_args = Args {
        args: args().skip(1).collect::<Vec<_>>(),
        envs: vars().collect::<_>(),
        cwd,
    };

    // Persist the captured args so thin builds can replay them later.
    write_rustc_args(&args_dir, &rustc_args);

    // Run the actual rustc command
    // We want all stdout/stderr to be inherited, so the running process can see the output
    //
    // Note that the args format we get from the wrapper includes the `rustc` command itself, so
    // we need to skip that - we already skipped the first arg when we created the args struct.
    let rustc = std::process::Command::new("rustc")
        .args(rustc_args.args.iter().skip(1))
        .envs(rustc_args.envs)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .current_dir(std::env::current_dir().expect("Failed to get current dir"))
        .status();

    // Propagate the exit code
    std::process::exit(rustc.unwrap().code().unwrap())
}

/// Write the captured rustc args to `{args_dir}/{crate_name}.{suffix}.json`.
///
/// The suffix is `lib` for lib/rlib crate types and `bin` otherwise. Crates without a
/// `--crate-name` argument (like the linker driver invocations) and cargo's `___` probe
/// crate are skipped.
fn write_rustc_args(args_dir: &Path, rustc_args: &Args) {
    let Some(crate_name) = rustc_args
        .args
        .iter()
        .skip_while(|arg| *arg != "--crate-name")
        .nth(1)
    else {
        return;
    };

    // A terrible hack to avoid writing non-sensical args when a build is completely fresh.
    if crate_name == "___" {
        return;
    }

    let crate_type: Option<&str> = rustc_args
        .args
        .iter()
        .skip_while(|arg| *arg != "--crate-type")
        .nth(1)
        .map(|s| s.as_str());
    let suffix = match crate_type {
        Some("lib" | "rlib") => "lib",
        _ => "bin",
    };

    // Drop the makeflags since they're tied to this specific cargo invocation and would
    // confuse a later replay of the captured args.
    let mut serialized = rustc_args.clone();
    serialized.envs.retain(|(key, _)| key != "CARGO_MAKEFLAGS");

    std::fs::create_dir_all(args_dir).expect("Failed to create rustc args dir");
    std::fs::write(
        args_dir.join(format!("{crate_name}.{suffix}.json")),
        serde_json::to_string(&serialized).expect("Failed to serialize rustc args"),
    )
    .expect("Failed to write rustc args to file");
}
