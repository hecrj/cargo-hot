use serde::{Deserialize, Serialize};
use std::{
    env::{args, vars},
    path::PathBuf,
};

/// The environment variable indicating where the args file is located.
///
/// When `dx-rustc` runs, it writes its arguments to this file.
pub const DX_RUSTC_WRAPPER_ENV_VAR: &str = "DX_RUSTC";

/// Is `dx` being used as a rustc wrapper?
///
/// This is primarily used to intercept cargo, enabling fast hot-patching by caching the environment
/// cargo setups up for the user's current project.
///
/// In a differenet world we could simply rely on cargo printing link args and the rustc command, but
/// it doesn't seem to output that in a reliable, parseable, cross-platform format (ie using command
/// files on windows...), so we're forced to do this interception nonsense.
pub fn is_wrapping_rustc() -> bool {
    std::env::var(DX_RUSTC_WRAPPER_ENV_VAR).is_ok()
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Args {
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
    /// it doesn't include first program name argument
    pub link_args: Vec<String>,
    /// The working directory the rustc process was invoked in. Thin builds replay rustc
    /// in this directory so relative paths in the captured args resolve the same way.
    #[serde(default)]
    pub cwd: PathBuf,
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

    let var_file: PathBuf = std::env::var(DX_RUSTC_WRAPPER_ENV_VAR)
        .expect("DX_RUSTC not set")
        .into();

    let cwd = std::env::current_dir().unwrap_or_default();

    let mut rustc_args = Args {
        args: args().skip(1).collect::<Vec<_>>(),
        envs: vars().collect::<_>(),
        link_args: Default::default(),
        cwd,
    };

    // A terrible hack to avoid writing non-sensical args when
    // a build is completely fresh.
    if rustc_args
        .args
        .iter()
        .skip_while(|arg| *arg != "--crate-name")
        .nth(1)
        .is_some_and(|name| name != "___")
    {
        rustc_args
            .envs
            .retain_mut(|(key, _)| key != "CARGO_MAKEFLAGS");

        std::fs::create_dir_all(var_file.parent().expect("Failed to get parent dir"))
            .expect("Failed to create parent dir");
        std::fs::write(
            &var_file,
            serde_json::to_string(&rustc_args).expect("Failed to serialize rustc args"),
        )
        .expect("Failed to write rustc args to file");
    }

    // Run the actual rustc command
    // We want all stdout/stderr to be inherited, so the running process can see the output
    //
    // Note that the args format we get from the wrapper includes the `rustc` command itself, so we
    // need to skip that - we already skipped the first arg when we created the args struct.
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
