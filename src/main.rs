use cargo_hot::Result;
use cargo_hot::hotpatch;
use cargo_hot::link::{self, LinkerFlavor};
use cargo_hot::rustc;
use cargo_hot_protocol::server;

use cargo::GlobalContext;
use cargo::core::{Target, TargetKind};
use cargo::util::{Filesystem, command_prelude::*};

use anyhow::{Context, anyhow, ensure};
use itertools::Itertools;
use serde::Deserialize;
use target_lexicon::{Architecture, OperatingSystem, Triple};
use tokio::process;
use tokio::sync::mpsc;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

#[tokio::main]
async fn main() -> Result<ExitCode> {
    #[cfg(debug_assertions)]
    tracing_subscriber::fmt::init();

    if rustc::is_wrapping_rustc() {
        rustc::run_rustc();
        return Ok(ExitCode::SUCCESS);
    }

    let gctx = GlobalContext::default()?;
    let _token = cargo::util::job::setup();

    let matches = Command::new("cargo")
        .bin_name("cargo")
        .subcommand(command())
        .try_get_matches()?;

    let (_, args) = matches
        .subcommand()
        .ok_or(anyhow!("`cargo-hot` must be called with `hot` subcommand"))?;

    let server = Server::new(gctx, args).await?;

    server.run().await
}

fn command() -> Command {
    subcommand("hot")
        .about("Run a binary or example of the local package in hot reloading mode")
        .arg(
            Arg::new("args")
                .value_name("ARGS")
                .help("Arguments for the binary or example to run")
                .value_parser(value_parser!(OsString))
                .num_args(0..)
                .trailing_var_arg(true),
        )
        .arg_message_format()
        .arg_silent_suggestion()
        .arg_package("Package with the target to run")
        .arg_targets_bin_example(
            "Name of the bin target to run",
            "Name of the example target to run",
        )
        .arg_features()
        .arg_parallel()
        .arg_release("Build artifacts in release mode, with optimizations")
        .arg_profile("Build artifacts with the specified profile")
        .arg_target_triple("Build for the target triple")
        .arg_target_dir()
        .arg_manifest_path()
        .arg_lockfile_path()
        .arg_ignore_rust_version()
        .arg_unit_graph()
        .arg_timings()
        .arg(
            opt(
                "verbose",
                "Use verbose output (-vv very verbose/build.rs output)",
            )
            .short('v')
            .action(ArgAction::Count)
            .global(true),
        )
        .after_help(color_print::cstr!(
            "Run `<cyan,bold>cargo help run</>` for more detailed information.\n"
        ))
}

#[derive(Debug)]
pub struct Server {
    gctx: GlobalContext,
    sysroot: PathBuf,
    crate_target: Target,
    crate_dir: PathBuf,
    workspace_dir: PathBuf,
    profile: String,
    triple: Triple,
    package: String,
    features: Vec<String>,
    exe_args: Vec<OsString>,
    extra_cargo_args: Vec<String>,
    extra_rustc_args: Vec<String>,
    verbose: u8,
    no_default_features: bool,
    target_dir: Filesystem,
    custom_linker: Option<PathBuf>,
    metadata: cargo_metadata::Metadata,
    /// Maps every file a crate depends on (from rustc dep-info files) to the crate that
    /// includes it. Change events are matched against this map so edits to non-`.rs`
    /// inputs (like `include_str!` targets or generated files) also trigger a patch.
    depinfo: std::sync::Mutex<HashMap<PathBuf, String>>,
}

#[derive(Clone, Debug)]
pub struct Build {
    exe: PathBuf,
    workspace_rustc: rustc::WorkspaceRustcArgs,
    time_start: SystemTime,
    patch_cache: Option<Arc<hotpatch::Cache>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuildMode {
    Fat,
    Thin {
        workspace_rustc: rustc::WorkspaceRustcArgs,
        /// Every crate modified since the fat build, including transitive workspace
        /// dependents. Each patch links the latest version of all of these crates.
        modified_crates: HashSet<String>,
        changed_files: Vec<PathBuf>,
        aslr_reference: u64,
        cache: Arc<hotpatch::Cache>,
    },
}

impl Server {
    async fn new(gctx: GlobalContext, args: &ArgMatches) -> Result<Self> {
        let sysroot = process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .await
            .map(|out| String::from_utf8(out.stdout).map(|s| s.trim().to_string()))?
            .context("Failed to extract rustc sysroot output")?;

        let target_kind = if args.contains_id("example") {
            TargetKind::ExampleBin
        } else {
            TargetKind::Bin
        };

        let workspace = args.workspace(&gctx)?;

        let compile_opts = args.compile_options(
            &gctx,
            CompileMode::Build,
            Some(&workspace),
            ProfileChecking::Custom,
        )?;

        let packages = compile_opts.spec.get_packages(&workspace)?;
        let main_package = packages.first().unwrap();

        let target_name = args
            .get_one("example")
            .cloned()
            .or(args.get_one("bin").cloned())
            .or_else(|| {
                if let Some(default_run) = &main_package.manifest().default_run() {
                    return Some(default_run.to_string());
                }

                let bin_count = packages
                    .iter()
                    .flat_map(|packages| packages.targets())
                    .filter(|target| target.kind() == &target_kind)
                    .count();

                if bin_count != 1 {
                    return None;
                }

                main_package.targets().iter().find_map(|x| {
                    if x.kind() == &target_kind {
                        Some(x.name().to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(main_package.name().to_string());

        let crate_target = main_package
            .targets()
            .iter()
            .find(|target| {
                target_name == target.name() && target.kind() == &target_kind
            })
            .with_context(|| {
                let target_of_kind = |kind|-> String {
                    let filtered_packages = main_package
                .targets()
                .iter()
                .filter_map(|target| {
                    (target.kind() == kind).then_some(target.name().to_string())
                }).collect::<Vec<_>>();

                filtered_packages.join(", ")};

                if let Some(example) = &args.get_one::<String>("example"){
                    let examples = target_of_kind(&TargetKind::ExampleBin);
                    format!("Failed to find example {example}. \nAvailable examples are:\n{examples}")
                } else if let Some(bin) = &args.get_one::<String>("bin") {
                    let binaries = target_of_kind(&TargetKind::Bin);
                    format!("Failed to find binary {bin}. \nAvailable binaries are:\n{binaries}")
                } else {
                    format!("Failed to find target {target_name}. \nIt looks like you are trying to build a library crate. \
                    You either need to run `cargo hot` from inside a binary crate or build a specific example with the `--example` flag. \
                    Available examples are:\n{}", target_of_kind(&TargetKind::ExampleBin))
                }
            })?
            .clone();

        let profile = match args.get_one::<String>("profile") {
            Some(profile) => profile.to_owned(),
            None if args.flag("release") => "release".to_string(),
            None => "dev".to_string(),
        };

        let triple = match args.get_one::<String>("target") {
            Some(target) => target.parse().expect("parse target"),
            None => target_lexicon::HOST,
        };

        // Determine the --package we'll pass to cargo.
        // todo: I think this might be wrong - we don't want to use main_package necessarily...
        let package = args
            .get_one("package")
            .cloned()
            .unwrap_or_else(|| main_package.name().to_string());

        let cargo_config = cargo_config2::Config::load().unwrap();

        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| cargo_config.build.target_dir.clone())
            .map(Filesystem::new)
            .unwrap_or_else(|| workspace.target_dir());

        let mut custom_linker = cargo_config.linker(triple.to_string())?;
        if let Some(linker) = custom_linker.as_ref()
            && (linker == "rust-lld" || linker == "rust-lld.exe")
            && cfg!(windows)
        {
            // When using "rust-lld.exe" as linker on windows, it still needs to have a flavor
            // given to it. rustc appears to be passing `-flavor "link"` when none is set by the
            // user. If no flavor is given, it fails with 'lld is a generic driver'.
            // We already use the existing lld-link by default on windows, so we can simply set the
            // `custom_linker` to `None` in these cases, since we end up using "lld-link" anyway
            // which is the same as "rust-lld.exe -flavor link".
            custom_linker = None;
        }

        let exe_args = args
            .get_many("args")
            .into_iter()
            .flatten()
            .cloned()
            .collect();

        let extra_cargo_args = vec![]; // TODO

        // TODO
        let extra_rustc_args = cargo_config
            .rustflags(triple.to_string())
            .unwrap_or_default()
            .unwrap_or_default()
            .flags;

        let crate_dir = main_package.manifest_path().parent().unwrap().to_path_buf();
        let workspace_dir = workspace.root_manifest().parent().unwrap().to_path_buf();

        // Resolve the workspace dependency graph so hotpatches can track which crates
        // are affected when a workspace member changes.
        let metadata = cargo_metadata::MetadataCommand::new()
            .current_dir(&workspace_dir)
            .exec()
            .context("Failed to read cargo metadata for the workspace")?;

        Ok(Self {
            gctx,
            sysroot: PathBuf::from(sysroot),
            crate_target,
            crate_dir,
            workspace_dir,
            profile,
            triple,
            package,
            features: args
                .get_many("features")
                .map(|features| features.into_iter().cloned().collect())
                .unwrap_or_default(),
            exe_args,
            extra_cargo_args,
            extra_rustc_args,
            verbose: args.get_count("verbose"),
            no_default_features: args.get_flag("no-default-features"),
            target_dir,
            custom_linker,
            metadata,
            depinfo: std::sync::Mutex::new(HashMap::new()),
        })
    }

    async fn run(self) -> Result<ExitCode> {
        use notify::Watcher;

        std::fs::create_dir_all(self.exe_dir())?;

        for file in [
            self.link_args_file(),
            self.link_err_file(),
            self.windows_command_file(),
        ] {
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(file)?;
        }

        // The rustc wrapper writes one json file per crate into this directory
        let _ = std::fs::create_dir_all(self.rustc_wrapper_args_dir());

        let (sender, mut receiver) = mpsc::channel(1);

        let handler = move |event: notify::Result<notify::Event>| {
            let Ok(event) = event else {
                return;
            };

            let is_allowed_notify_event = match event.kind {
                notify::EventKind::Modify(notify::event::ModifyKind::Data(_)) => true,
                notify::EventKind::Modify(notify::event::ModifyKind::Name(_)) => true,
                // The primary modification event on WSL's poll watcher.
                notify::EventKind::Modify(notify::event::ModifyKind::Metadata(
                    notify::event::MetadataKind::WriteTime,
                )) => true,
                // Catch-all for unknown event types (windows)
                notify::EventKind::Modify(notify::event::ModifyKind::Any) => true,
                notify::EventKind::Modify(notify::event::ModifyKind::Metadata(_)) => false,
                // Don't care about anything else.
                notify::EventKind::Create(_) => true,
                notify::EventKind::Remove(_) => true,
                _ => false,
            };

            if is_allowed_notify_event {
                let _ = sender.blocking_send(event);
            }
        };

        let mut watcher: Box<dyn Watcher> = if is_wsl() {
            // On wsl, we need to poll the filesystem for changes
            Box::new(notify::PollWatcher::new(
                handler,
                notify::Config::default().with_poll_interval(Duration::from_secs(2)),
            )?)
        } else {
            // Otherwise we can use the recommended watcher
            Box::new(notify::recommended_watcher(handler)?)
        };

        let build = self.build(BuildMode::Fat).await?;
        let mut executable = tokio::process::Command::new(self.main_exe())
            .args(&self.exe_args)
            .spawn()?;

        // Watch every workspace member so edits to dependency crates trigger hotpatches
        for member_dir in self.workspace_member_dirs() {
            watcher.watch(&member_dir, notify::RecursiveMode::Recursive)?;
        }

        let mut server = server::Server::bind().await?;
        let mut connection = server.accept().await?;
        let mut buffer = Vec::new();

        // Every crate modified since the fat build. Each patch is self-contained and links
        // the latest version of *all* of these crates, not just the one that changed in the
        // current iteration.
        let mut modified_crates: HashSet<String> = HashSet::new();
        let _ = modified_crates.insert(self.tip_package_name());

        loop {
            let n = tokio::select! {
                n = read_batch(&mut receiver, &mut buffer, Duration::from_millis(100)) => n,
                status = executable.wait() => {
                    return Ok(status?.code().map(|code| ExitCode::from(code as u8)).unwrap_or_default())
                }
            };

            if n == 0 {
                return Err(anyhow!("file notifier failed"));
            }

            // Collect the changed files and attribute them to crates. The dep-info filemap is
            // the precise source (it covers non-`.rs` inputs like `include_str!` targets, even
            // ones under `target/`); the workspace-dir heuristic below is the fallback for
            // files the filemap doesn't know yet.
            let mut changed_files: BTreeSet<PathBuf> = BTreeSet::new();
            let mut changed_crates: HashSet<String> = HashSet::new();
            let mut heuristic_files: BTreeSet<PathBuf> = BTreeSet::new();

            {
                let depinfo = self.depinfo.lock().unwrap();
                for event in buffer.drain(..n) {
                    for path in event.paths {
                        if let Some(crate_name) = depinfo.get(&path) {
                            let _ = changed_files.insert(path);
                            let _ = changed_crates.insert(crate_name.clone());
                        } else if path == path.with_extension("rs")
                            && // Ignore anything under a `target` directory - a member whose
                               // source dir is the workspace root would otherwise pick up the
                               // whole target tree.
                            !path.components().any(|component| {
                                matches!(
                                    component,
                                    std::path::Component::Normal(name) if name == "target"
                                )
                            })
                        {
                            let _ = changed_files.insert(path.clone());
                            let _ = heuristic_files.insert(path);
                        }
                    }
                }
            }

            // Files the filemap doesn't know yet resolve to the workspace member whose
            // crate directory contains them.
            for file in &heuristic_files {
                if let Some(crate_name) = self.file_to_workspace_crate(file) {
                    let _ = changed_crates.insert(crate_name);
                }
            }

            if changed_files.is_empty() {
                continue;
            }

            // Expand the cumulative modified set with the workspace dependents of each
            // changed crate. Only crates that cascade to the tip matter - the rest are
            // unrelated to this binary and were never built by the fat build.
            let tip_package_name = self.tip_package_name();
            let mut reaches_tip = false;
            for crate_name in &changed_crates {
                if *crate_name == tip_package_name {
                    reaches_tip = true;
                    continue;
                }

                let closure = self.workspace_dependent_closure(crate_name);
                if !closure.contains(&tip_package_name) {
                    continue;
                }

                reaches_tip = true;
                for dependent in closure {
                    if dependent != tip_package_name {
                        let _ = modified_crates.insert(dependent);
                    }
                }
            }

            if !reaches_tip {
                log::debug!(
                    "Changes to {changed_crates:?} do not affect `{tip_package_name}`; \
                     skipping patch"
                );
                continue;
            }

            let _ = self.gctx.shell().status(
                "Patching",
                format!(
                    "{} ({})",
                    self.crate_target.name(),
                    self.crate_dir.display(),
                ),
            );

            let start = Instant::now();

            match self
                .patch(&build, changed_files, &modified_crates, &mut connection)
                .await
            {
                Ok(()) => {
                    let _ = self.gctx.shell().status(
                        "Finished",
                        format!(
                            "`{}` profile target(s) in {:.2}s",
                            self.profile,
                            start.elapsed().as_millis() as f32 / 1_000.0
                        ),
                    );
                }
                Err(error) => {
                    log::error!("{error}");
                }
            }
        }
    }

    async fn build(&self, mode: BuildMode) -> Result<Build> {
        // Before a fat build, make sure every workspace member in the tip's tree gets
        // captured by the rustc wrapper (cleaning any that are missing a capture).
        if matches!(mode, BuildMode::Fat) {
            self.ensure_workspace_captures().await?;
        }

        // Rebuild the modified workspace dependency crates before recompiling the tip, so the
        // tip's codegen reads their fresh rlibs and the patch can link the new code in.
        if let BuildMode::Thin {
            workspace_rustc,
            modified_crates,
            ..
        } = &mode
        {
            let replayed_crates = self.workspace_hotpatch_replay_order(modified_crates)?;
            for crate_name in &replayed_crates {
                let Some(rustc_args) =
                    self.workspace_hotpatch_replay_args(workspace_rustc, crate_name)
                else {
                    // The crate isn't actually built by the fat build (e.g. it only reaches
                    // the tip through a disabled feature) - nothing to replay
                    log::debug!("Skipping workspace crate '{crate_name}': no captured rustc args");
                    continue;
                };

                log::debug!("Replaying workspace crate '{crate_name}'");

                self.compile_dep_crate(crate_name, &rustc_args)
                    .await
                    .with_context(|| format!("Failed to replay workspace crate '{crate_name}'"))?;

                // rustc just rewrote this crate's dep-info - fold its inputs into the
                // filemap so edits to its `include_str!` targets and co. trigger a patch.
                self.refresh_dep_info(crate_name, &rustc_args, true);
            }
        }

        // Run the cargo build to produce our artifacts. For fat builds the filemap starts
        // fresh - it gets repopulated from the dep-info of every member below.
        if matches!(mode, BuildMode::Fat) {
            self.depinfo.lock().unwrap().clear();
        }

        let mut build = self.cargo_build(&mode).await?;

        // Write the build artifacts to the bundle on the disk
        match &mode {
            BuildMode::Thin {
                aslr_reference,
                cache,
                modified_crates,
                ..
            } => {
                self.write_patch(*aslr_reference, &mut build, cache, modified_crates)
                    .await?;

                // The tip was just recompiled - rustc rewrote its dep-info, so fold any new
                // inputs (e.g. a fresh `include_str!` target) into the filemap. Existing
                // entries win so shared files still trigger the dependency's replay.
                let tip_key = format!("{}.bin", self.tip_crate_name());
                if let Some(tip_args) = build.workspace_rustc.rustc_args.get(&tip_key) {
                    self.refresh_dep_info(&self.tip_crate_name(), tip_args, false);
                }
            }

            BuildMode::Fat => {
                // Sync the executable into the exe dir. This can be needed even for fresh
                // builds - a previous run may have built the artifact but failed to copy
                // it (e.g. the old executable was still running: ETXTBSY).
                let exe_mtime = std::fs::metadata(&build.exe).and_then(|meta| meta.modified());
                let main_mtime =
                    std::fs::metadata(self.main_exe()).and_then(|meta| meta.modified());

                let needs_copy = match (exe_mtime, main_mtime) {
                    (Ok(exe_mtime), Ok(main_mtime)) => exe_mtime > main_mtime,
                    _ => true,
                };

                if needs_copy {
                    self.write_executable(&build.exe)
                        .await
                        .context("Failed to write main executable")?;

                    log::debug!("Binary created at {}", self.build_dir().display());
                }

                // Populate the filemap from the dep-info of every captured workspace member.
                // The dep-info files on disk are current in either case: recompiled members
                // just had theirs rewritten, and fresh members' paths are unchanged.
                //
                // The tip is registered last so that a file shared between the tip and a
                // dependency keeps the dependency's entry - editing it then triggers the
                // dependency's replay, whose closure also rebuilds the tip.
                let tip_crate = self.tip_crate_name();

                for (key, args) in build.workspace_rustc.rustc_args.iter() {
                    let Some((crate_name, _)) = key.rsplit_once('.') else {
                        continue;
                    };

                    if *crate_name != tip_crate {
                        self.refresh_dep_info(crate_name, args, false);
                    }
                }

                for suffix in ["lib", "bin"] {
                    if let Some(args) = build
                        .workspace_rustc
                        .rustc_args
                        .get(&format!("{tip_crate}.{suffix}"))
                    {
                        self.refresh_dep_info(&tip_crate, args, false);
                    }
                }
            }
        }

        // Populate the patch cache if we're in fat mode
        if matches!(mode, BuildMode::Fat) {
            build.patch_cache = Some(Arc::new(self.create_patch_cache(&build.exe).await?));
        }

        Ok(build)
    }

    async fn patch(
        &self,
        build: &Build,
        changed_files: BTreeSet<PathBuf>,
        modified_crates: &HashSet<String>,
        connection: &mut server::Connection,
    ) -> Result<()> {
        let patch = self
            .build(BuildMode::Thin {
                workspace_rustc: build.workspace_rustc.clone(),
                modified_crates: modified_crates.clone(),
                changed_files: changed_files.into_iter().collect(),
                aslr_reference: connection.aslr_reference() as u64,
                cache: build.patch_cache.clone().unwrap(),
            })
            .await?;

        let jump_table = hotpatch::create_jump_table(
            &self.patch_exe(patch.time_start),
            &self.triple,
            build.patch_cache.as_ref().unwrap(),
        )?;

        if jump_table.map.is_empty() {
            log::warn!(
                "Hot-patch jump table is empty - the patch will have no effect. This usually \
                 means no symbols in the patch matched the running binary."
            );
        }

        connection.patch(&jump_table).await?;

        Ok(())
    }

    async fn create_patch_cache(&self, exe: &Path) -> Result<hotpatch::Cache> {
        // TODO: Wasm
        let exe = exe.to_path_buf();

        Ok(hotpatch::Cache::new(&exe, &self.triple)?)
    }

    async fn write_executable(&self, exe: &Path) -> Result<()> {
        // TODO: Wasm
        let _ = std::fs::copy(exe, self.main_exe())?;

        Ok(())
    }

    /// Run the cargo build by assembling the build command and executing it.
    ///
    /// This method needs to be very careful with processing output since errors being swallowed will
    /// be very confusing to the user.
    async fn cargo_build(&self, mode: &BuildMode) -> Result<Build> {
        use tokio::io::AsyncBufReadExt;

        let time_start = SystemTime::now();
        let mut cmd = self.build_command(mode)?;

        log::debug!("Executing cargo for {}", self.triple);

        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("Failed to spawn cargo build")?;

        let stdout = tokio::io::BufReader::new(child.stdout.take().unwrap());
        let stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
        let mut output_location: Option<PathBuf> = None;
        let mut stdout = stdout.lines();
        let mut stderr = stderr.lines();
        let mut emitting_error = false;
        let mut has_compiled = false;

        // TODO
        // let mut units_compiled = 0;

        loop {
            use cargo_metadata::Message;
            use cargo_metadata::diagnostic::Diagnostic;

            let line = tokio::select! {
                Ok(Some(line)) = stdout.next_line() => line,
                Ok(Some(line)) = stderr.next_line() => line,
                else => break,
            };

            let Some(Ok(message)) = Message::parse_stream(std::io::Cursor::new(&line)).next()
            else {
                continue;
            };

            match message {
                Message::BuildScriptExecuted(_) => {
                    // TODO
                    // units_compiled += 1;
                }
                Message::CompilerMessage(msg) => println!("{}", msg.message),
                Message::TextLine(line) => {
                    // Handle the case where we're getting lines directly from rustc.
                    // These are in a different format than the normal cargo output, though I imagine
                    // this parsing code is quite fragile/sensitive to changes in cargo, cargo_metadata, rustc, etc.
                    #[derive(Deserialize)]
                    struct RustcArtifact {
                        artifact: PathBuf,
                        emit: String,
                    }

                    // These outputs look something like:
                    //
                    // { "artifact":"target/debug/deps/libdioxus_core-4f2a0b3c1e5f8b7c.rlib", "emit":"link" }
                    //
                    // There are other outputs like depinfo that we might be interested in in the future.
                    if let Ok(artifact) = serde_json::from_str::<RustcArtifact>(&line) {
                        if artifact.emit == "link" {
                            output_location = Some(artifact.artifact);
                        }

                        continue;
                    }

                    // Handle direct rustc diagnostics
                    if let Ok(diag) = serde_json::from_str::<Diagnostic>(&line) {
                        if let Some(rendered) = diag.rendered {
                            println!("{rendered}");
                        }

                        continue;
                    }

                    // For whatever reason, if there's an error while building, we still receive the TextLine
                    // instead of an "error" message. However, the following messages *also* tend to
                    // be the error message, and don't start with "error:". So we'll check if we've already
                    // emitted an error message and if so, we'll emit all following messages as errors too.
                    //
                    // todo: This can lead to some really ugly output though, so we might want to look
                    // into a more reliable way to detect errors propagating out of the compiler. If
                    // we always wrapped rustc, then we could store this data somewhere in a much more
                    // reliable format.
                    if line.trim_start().starts_with("error:") {
                        emitting_error = true;
                    }

                    // Note that previous text lines might have set emitting_error to true
                    match emitting_error {
                        true => eprintln!("{line}"),
                        false => println!("{line}"),
                    }
                }
                Message::CompilerArtifact(artifact) => {
                    // TODO
                    // units_compiled += 1;
                    has_compiled = has_compiled || !artifact.fresh;
                    output_location = artifact.executable.map(Into::into);
                }
                // todo: this can occasionally swallow errors, so we should figure out what exactly is going wrong
                //       since that is a really bad user experience.
                Message::BuildFinished(finished) if !finished.success => {
                    return Err(anyhow::anyhow!(
                        "Cargo build failed, signaled by the compiler. Toggle tracing mode (press `t`) for more information."
                    ));
                }
                _ => {}
            }
        }

        // Accumulate the rustc args the wrapper captured for every crate in the build
        let workspace_rustc = self.load_rustc_argset()?;

        // If there's any warnings from the linker, we should print them out
        if let Ok(linker_warnings) = std::fs::read_to_string(self.link_err_file())
            && !linker_warnings.is_empty()
        {
            if output_location.is_none() {
                log::error!("Linker warnings: {linker_warnings}");
            } else {
                log::debug!("Linker warnings: {linker_warnings}");
            }
        }

        let exe = output_location.context("Cargo build failed - no output location. Toggle tracing mode (press `t`) for more information.")?;

        // Fat builds need to be linked with the fat linker. Would also like to link here for thin builds
        if matches!(mode, BuildMode::Fat) && has_compiled {
            let link_start = SystemTime::now();
            self.run_fat_link(&exe, &workspace_rustc).await?;

            log::debug!(
                "Fat linking completed in {}us",
                SystemTime::now()
                    .duration_since(link_start)
                    .unwrap()
                    .as_micros()
            );
        }

        let time_end = SystemTime::now();

        log::debug!(
            "Build completed successfully in {}us: {:?}",
            time_end.duration_since(time_start).unwrap().as_micros(),
            exe
        );

        Ok(Build {
            exe,
            workspace_rustc,
            time_start,
            patch_cache: None,
        })
    }

    async fn write_patch(
        &self,
        aslr_reference: u64,
        build: &mut Build,
        cache: &Arc<hotpatch::Cache>,
        modified_crates: &HashSet<String>,
    ) -> anyhow::Result<()> {
        log::debug!(
            "Original builds for patch: {}",
            self.link_args_file().display()
        );

        let raw_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?;

        let args = raw_args.lines().collect::<Vec<_>>();

        // The captured args of the tip crate's bin target, used for the linker environment
        let tip_crate_key = format!("{}.bin", self.tip_crate_name());
        let tip_rustc_args = build
            .workspace_rustc
            .rustc_args
            .get(&tip_crate_key)
            .with_context(|| {
                format!("Missing captured rustc args for the tip crate '{tip_crate_key}'")
            })?;

        // Include the rlibs of every replayed workspace crate in the patch. Their code changed
        // since the fat build, so it must be linked in instead of being resolved via stubs.
        let replayed_crates = self.workspace_hotpatch_replay_order(modified_crates)?;
        let workspace_rlibs =
            self.workspace_hotpatch_link_rlibs(&build.workspace_rustc, &replayed_crates)?;
        log::debug!("Workspace rlibs for patch: {workspace_rlibs:?}");

        // Extract out the incremental object files.
        //
        // This is sadly somewhat of a hack, but it might be a moderately reliable hack.
        //
        // When rustc links your project, it passes the args as how a linker would expect, but with
        // a somewhat reliable ordering. These are all internal details to cargo/rustc, so we can't
        // rely on them *too* much, but the *are* fundamental to how rust compiles your projects, and
        // linker interfaces probably won't change drastically for another 40 years.
        //
        // We need to tear apart this command and only pass the args that are relevant to our thin link.
        // Mainly, we don't want any rlibs to be linked. Occasionally some libraries like objc_exception
        // export a folder with their artifacts - unsure if we actually need to include them. Generally
        // you can err on the side that most *libraries* don't need to be linked here since dlopen
        // satisfies those symbols anyways when the binary is loaded.
        //
        // Many args are passed twice, too, which can be confusing, but generally don't have any real
        // effect. Note that on macos/ios, there's a special macho header that needs to be set, otherwise
        // dyld will complain.
        //
        // Also, some flags in darwin land might become deprecated, need to be super conservative:
        // - https://developer.apple.com/forums/thread/773907
        //
        // The format of this command roughly follows:
        // ```
        // clang
        //     /dioxus/target/debug/subsecond-cli
        //     /var/folders/zs/gvrfkj8x33d39cvw2p06yc700000gn/T/rustcAqQ4p2/symbols.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.05stnb4bovskp7a00wyyf7l9s.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.08rgcutgrtj2mxoogjg3ufs0g.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.0941bd8fa2bydcv9hfmgzzne9.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/libbincode-c215feeb7886f81b.rlib
        //     /dioxus/target/subsecond-dev/deps/libanyhow-e69ac15c094daba6.rlib
        //     /dioxus/target/subsecond-dev/deps/libratatui-c3364579b86a1dfc.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libstd-019f0f6ae6e6562b.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libpanic_unwind-7387d38173a2eb37.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libobject-2b03cf6ece171d21.rlib
        //     -framework AppKit
        //     -lc
        //     -framework Foundation
        //     -framework Carbon
        //     -lSystem
        //     -framework CoreFoundation
        //     -lobjc
        //     -liconv
        //     -lm
        //     -arch arm64
        //     -mmacosx-version-min=11.0.0
        //     -L /dioxus/target/subsecond-dev/build/objc_exception-dc226cad0480ea65/out
        //     -o /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa
        //     -nodefaultlibs
        //     -Wl,-all_load
        // ```
        let mut dylibs = vec![];
        let mut object_files = args
            .iter()
            .filter(|arg| arg.ends_with(".rcgu.o"))
            .sorted()
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        object_files.extend(workspace_rlibs);

        // On non-wasm platforms, we generate a special shim object file which converts symbols from
        // fat binary into direct addresses from the running process.
        //
        // Our wasm approach is quite specific to wasm. We don't need to resolve any missing symbols
        // there since wasm is relocatable, but there is considerable pre and post processing work to
        // satisfy undefined symbols that we do by munging the binary directly.
        //
        // todo: can we adjust our wasm approach to also use a similar system?
        // todo: don't require the aslr reference and just patch the got when loading.
        //
        // Requiring the ASLR offset here is necessary but unfortunately might be flakey in practice.
        // Android apps can take a long time to open, and a hot patch might've been issued in the interim,
        // making this hotpatch a failure.
        if !is_wasm_or_wasi(&self.triple) {
            let stub_bytes = hotpatch::create_undefined_symbol_stub(
                cache,
                &object_files,
                &self.triple,
                aslr_reference,
            )
            .expect("failed to resolve patch symbols");

            // Currently we're dropping stub.o in the exe dir, but should probably just move to a tempfile?
            let patch_file = self.main_exe().with_file_name("stub.o");
            std::fs::write(&patch_file, stub_bytes)?;
            object_files.push(patch_file);

            // Add the dylibs/sos to the linker args
            // Make sure to use the one in the bundle, not the ones in the target dir or system.
            for arg in args.iter() {
                if arg.ends_with(".dylib") || arg.ends_with(".so") {
                    let path = PathBuf::from(arg);
                    dylibs.push(self.frameworks_folder().join(path.file_name().unwrap()));
                }
            }
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;
        let out_exe = self.patch_exe(build.time_start);
        let out_arg = match self.triple.operating_system {
            OperatingSystem::Windows => vec![format!("/OUT:{}", out_exe.display())],
            _ => vec!["-o".to_string(), out_exe.display().to_string()],
        };

        log::trace!("Linking with {linker:?} using args: {object_files:#?}");

        let mut out_args: Vec<OsString> = vec![];
        out_args.extend(object_files.iter().map(Into::into));
        out_args.extend(dylibs.iter().map(Into::into));
        out_args.extend(self.thin_link_args(&args)?.iter().map(Into::into));
        out_args.extend(out_arg.iter().map(Into::into));

        if cfg!(windows) {
            let cmd_contents: String = out_args
                .iter()
                .map(|s| format!("\"{}\"", s.to_string_lossy()))
                .join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display()).into()];
        }

        // Add more search paths for the linker
        let mut command_envs: Vec<(String, String)> = tip_rustc_args.envs.clone();

        // On linux, we need to set a more complete PATH for the linker to find its libraries
        if cfg!(target_os = "linux") {
            command_envs.push(("PATH".to_string(), std::env::var("PATH").unwrap()));
        }

        // Run the linker directly!
        //
        // We dump its output directly into the patch exe location which is different than how rustc
        // does it since it uses llvm-objcopy into the `target/debug/` folder.
        let res = tokio::process::Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(command_envs.iter().map(|(k, v)| (k, v)))
            .output()
            .await?;

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            if !self.patch_exe(build.time_start).exists() || !res.status.success() {
                log::error!("Failed to generate patch: {}", errs.trim());
            } else {
                log::trace!("Linker output during thin linking: {}", errs.trim());
            }
        }

        // For some really weird reason that I think is because of dlopen caching, future loads of the
        // jump library will fail if we don't remove the original fat file. I think this could be
        // because of library versioning and namespaces, but really unsure.
        //
        // The errors if you forget to do this are *extremely* cryptic - missing symbols that never existed.
        //
        // Fortunately, this binary exists in two places - the deps dir and the target out dir. We
        // can just remove the one in the deps dir and the problem goes away.
        if let Some(idx) = args.iter().position(|arg| *arg == "-o") {
            _ = std::fs::remove_file(PathBuf::from(args[idx + 1]));
        }

        // Clean up the temps manually
        // todo: we might want to keep them around for debugging purposes
        //
        // The workspace rlibs are left in place - they're the real cargo outputs and the next
        // patch reuses them.
        for file in object_files.iter().filter(|file| !file.ends_with(".rlib")) {
            _ = std::fs::remove_file(file);
        }

        Ok(())
    }

    /// Patches are stored in the same directory as the main executable, but with a name based on the
    /// time the patch started compiling.
    ///
    /// - lib{name}-patch-{time}.(so/dll/dylib) (next to the main exe)
    ///
    /// Note that weirdly enough, the name of dylibs can actually matter. In some environments, libs
    /// can override each other with symbol interposition.
    ///
    /// Also, on Android - and some Linux, we *need* to start the lib name with `lib` for the dynamic
    /// loader to consider it a shared library.
    ///
    /// todo: the time format might actually be problematic if two platforms share the same build folder.
    fn patch_exe(&self, time_start: SystemTime) -> PathBuf {
        let path = self.main_exe().with_file_name(format!(
            "lib{}-patch-{}",
            self.executable_name(),
            time_start
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|f| f.as_millis())
                .unwrap_or(0),
        ));

        let extension = match self.linker_flavor() {
            LinkerFlavor::Darwin => "dylib",
            LinkerFlavor::Gnu => "so",
            LinkerFlavor::WasmLld => "wasm",
            LinkerFlavor::Msvc => "dll",
            LinkerFlavor::Unsupported => "",
        };

        path.with_extension(extension)
    }

    /// Take the original args passed to the "fat" build and then create the "thin" variant.
    ///
    /// This is basically just stripping away the rlibs and other libraries that will be satisfied
    /// by our stub step.
    fn thin_link_args(&self, original_args: &[&str]) -> Result<Vec<String>> {
        let mut out_args = vec![];

        match self.linker_flavor() {
            // wasm32-unknown-unknown -> use wasm-ld (gnu-lld)
            //
            // We need to import a few things - namely the memory and ifunc table.
            //
            // We can safely export everything, I believe, though that led to issues with the "fat"
            // binaries that also might lead to issues here too. wasm-bindgen chokes on some symbols
            // and the resulting JS has issues.
            //
            // We turn on both --pie and --experimental-pic but I think we only need --pie.
            //
            // We don't use *any* of the original linker args since they do lots of custom exports
            // and other things that we don't need.
            //
            // The trickiest one here is -Crelocation-model=pic, which forces data symbols
            // into a GOT, making it possible to import them from the main module.
            //
            // I think we can make relocation-model=pic work for non-wasm platforms, enabling
            // fully relocatable modules with no host coordination in lieu of sending out
            // the aslr slide at runtime.
            LinkerFlavor::WasmLld => {
                out_args.extend([
                    "--fatal-warnings".to_string(),
                    "--verbose".to_string(),
                    "--import-memory".to_string(),
                    "--import-table".to_string(),
                    "--growable-table".to_string(),
                    "--export".to_string(),
                    "main".to_string(),
                    "--allow-undefined".to_string(),
                    "--no-demangle".to_string(),
                    "--no-entry".to_string(),
                    "--pie".to_string(),
                    "--experimental-pic".to_string(),
                ]);

                // retain exports so post-processing has hooks to work with
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "--export" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }
                }
            }

            // This uses "cc" and these args need to be ld compatible
            //
            // Most importantly, we want to pass `-dylib` to both CC and the linker to indicate that
            // we want to generate the shared library instead of an executable.
            LinkerFlavor::Darwin => {
                out_args.extend(["-Wl,-dylib".to_string()]);

                // Preserve the original args. We only preserve:
                // -framework
                // -arch
                // -L <path>
                // -lxyz
                // -m (arch/emulation)
                // -target
                // -isysroot (iOS only)
                // -nodefaultlibs
                // -fuse-ld (linker selection)
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-framework"
                        || *arg == "-arch"
                        || *arg == "-L"
                        || *arg == "-target"
                        || (*arg == "-isysroot"
                            && matches!(self.triple.operating_system, OperatingSystem::IOS(_)))
                    {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-Wl,-fuse-ld")
                        || arg.starts_with("-fuse-ld")
                        || arg.starts_with("-nodefaultlibs")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            // android/linux need to be compatible with lld
            //
            // android currently drags along its own libraries and other zany flags
            LinkerFlavor::Gnu => {
                out_args.extend([
                    "-shared".to_string(),
                    "-Wl,--eh-frame-hdr".to_string(),
                    "-Wl,-z,noexecstack".to_string(),
                    "-Wl,-z,relro,-z,now".to_string(),
                    "-nodefaultlibs".to_string(),
                    "-Wl,-Bdynamic".to_string(),
                ]);

                // Preserve the original args. We only preserve:
                // -L <path>
                // -lxyz
                // -m (arch/emulation)
                // -B<path>  (gcc program search path — Rust 1.86+ injects -B/gcc-ld + -fuse-ld=lld
                //            so that cc picks up the bundled lld; we must forward it for the patch
                //            linker invocation too, otherwise cc falls back to the system `ld`)
                // -fuse-ld  (linker selection)
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-L" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-Wl,--target=")
                        || arg.starts_with("-Wl,-fuse-ld")
                        || arg.starts_with("-fuse-ld")
                        || arg.starts_with("-B")
                        || arg.contains("-ld-path")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            LinkerFlavor::Msvc => {
                out_args.extend([
                    "shlwapi.lib".to_string(),
                    "kernel32.lib".to_string(),
                    "advapi32.lib".to_string(),
                    "ntdll.lib".to_string(),
                    "userenv.lib".to_string(),
                    "ws2_32.lib".to_string(),
                    "dbghelp.lib".to_string(),
                    "/defaultlib:msvcrt".to_string(),
                    "/DLL".to_string(),
                    "/DEBUG".to_string(),
                    "/PDBALTPATH:%_PDB%".to_string(),
                    "/EXPORT:main".to_string(),
                    "/HIGHENTROPYVA:NO".to_string(),
                ]);
            }

            LinkerFlavor::Unsupported => {
                return Err(anyhow::anyhow!("Unsupported platform for thin linking"));
            }
        }

        let extract_value = |arg: &str| -> Option<String> {
            original_args
                .iter()
                .position(|a| *a == arg)
                .map(|i| original_args[i + 1].to_string())
        };

        if let Some(vale) = extract_value("-target") {
            out_args.push("-target".to_string());
            out_args.push(vale);
        }

        if let Some(vale) = extract_value("-isysroot")
            && matches!(self.triple.operating_system, OperatingSystem::IOS(_))
        {
            out_args.push("-isysroot".to_string());
            out_args.push(vale);
        }

        Ok(out_args)
    }

    fn main_exe(&self) -> PathBuf {
        self.exe_dir().join(self.platform_exe_name())
    }

    fn executable_name(&self) -> &str {
        self.crate_target.name()
    }

    fn platform_exe_name(&self) -> String {
        if self.triple.operating_system == OperatingSystem::Windows {
            return format!("{}.exe", self.executable_name());
        }

        if self.triple.architecture == Architecture::Wasm32
            || self.triple.architecture == Architecture::Wasm64
        {
            // this will be wrong, I think, but not important?
            return format!("{}_bg.wasm", self.executable_name());
        }

        self.executable_name().to_string()
    }

    fn exe_dir(&self) -> PathBuf {
        self.build_dir()
    }

    fn build_dir(&self) -> PathBuf {
        self.internal_out_dir()
            .join(self.executable_name())
            .join(&self.profile)
    }

    fn internal_out_dir(&self) -> PathBuf {
        self.target_dir.as_path_unlocked().join("cargo-hot")
    }

    /// When we link together the fat binary, we need to make sure every `.o` file in *every* rlib
    /// is taken into account. This is the same work that the rust compiler does when assembling
    /// staticlibs.
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_codegen_ssa/src/back/link.rs#L448>
    ///
    /// Since we're going to be passing these to the linker, we need to make sure and not provide any
    /// weird files (like the rmeta) file that rustc generates.
    ///
    /// We discovered the need for this after running into issues with wasm-ld not being able to
    /// handle the rmeta file.
    ///
    /// <https://github.com/llvm/llvm-project/issues/55786>
    ///
    /// Also, crates might not drag in all their dependent code. The monorphizer won't lift trait-based generics:
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_monomorphize/src/collector.rs>
    ///
    /// When Rust normally handles this, it uses the +whole-archive directive which adjusts how the rlib
    /// is written to disk.
    ///
    /// Since creating this object file can be a lot of work, we cache it in the target dir by hashing
    /// the names of the rlibs in the command and storing it in the target dir. That way, when we run
    /// this command again, we can just used the cached object file.
    ///
    /// In theory, we only need to do this for every crate accessible by the current crate, but that's
    /// hard acquire without knowing the exported symbols from each crate.
    ///
    /// todo: I think we can traverse our immediate dependencies and inspect their symbols, unless they `pub use` a crate
    /// todo: we should try and make this faster with memmapping
    pub(crate) async fn run_fat_link(
        &self,
        exe: &Path,
        workspace_rustc_args: &rustc::WorkspaceRustcArgs,
    ) -> Result<()> {
        use uuid::Uuid;

        // Get the rustc args of the tip crate's bin target, used for the linker environment
        let rustc_args = workspace_rustc_args
            .rustc_args
            .get(&format!("{}.bin", self.tip_crate_name()))
            .context("Missing rustc capture for the tip crate")?;

        ensure!(
            !workspace_rustc_args.link_args.is_empty(),
            "Missing linker args for the fat link of '{}'. The tip crate likely did not run through linker interception for this build.",
            self.tip_crate_name()
        );

        // Filter out the rlib files from the arguments
        let rlibs = workspace_rustc_args
            .link_args
            .iter()
            .filter(|arg| arg.ends_with(".rlib"))
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        // Acquire a hash from the rlib names, sizes, modified times, and dx's git commit hash
        // This ensures that any changes in dx or the rlibs will cause a new hash to be generated
        // The hash relies on both dx and rustc hashes, so it should be thoroughly unique. Keep it
        // short to avoid long file names.
        let hash_id = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            rlibs
                .iter()
                .map(|p| {
                    format!(
                        "{}-{}-{}-{}",
                        p.file_name().unwrap().to_string_lossy(),
                        p.metadata().map(|m| m.len()).unwrap_or_default(),
                        p.metadata()
                            .ok()
                            .and_then(|m| m.modified().ok())
                            .and_then(|f| f
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .map(|f| f.as_secs())
                                .ok())
                            .unwrap_or_default(),
                        env!("CARGO_PKG_VERSION"),
                    )
                })
                .collect::<String>()
                .as_bytes(),
        )
        .to_string()
        .chars()
        .take(8)
        .collect::<String>();

        // Check if we already have a cached object file
        let out_ar_path = exe.with_file_name(format!("libdeps-{hash_id}.a",));
        let out_rlibs_list = exe.with_file_name(format!("rlibs-{hash_id}.txt"));
        let mut archive_has_contents = out_ar_path.exists();

        // Use the rlibs list if it exists
        let mut compiler_rlibs = std::fs::read_to_string(&out_rlibs_list)
            .ok()
            .map(|s| s.lines().map(PathBuf::from).collect::<Vec<_>>())
            .unwrap_or_default();

        // Create it by dumping all the rlibs into it
        // This will include the std rlibs too, which can severely bloat the size of the archive
        //
        // The nature of this process involves making extremely fat archives, so we should try and
        // speed up the future linking process by caching the archive.
        //
        // Since we're using the git hash for the CLI entropy, debug builds should always regenerate
        // the archive since their hash might not change, but the logic might.
        if !archive_has_contents || cfg!(debug_assertions) {
            compiler_rlibs.clear();

            let mut bytes = vec![];
            let mut out_ar = ar::Builder::new(&mut bytes);
            for rlib in &rlibs {
                // Skip compiler rlibs since they're missing bitcode
                //
                // https://github.com/rust-lang/rust/issues/94232#issuecomment-1048342201
                //
                // if the rlib is not in the target directory, we skip it.
                if !rlib.starts_with(&self.workspace_dir) {
                    compiler_rlibs.push(rlib.clone());
                    log::trace!("Skipping rlib: {rlib:?}");
                    continue;
                }

                log::trace!("Adding rlib to staticlib: {rlib:?}");

                let rlib_contents = std::fs::read(rlib)?;
                let mut reader = ar::Archive::new(std::io::Cursor::new(rlib_contents));
                let mut keep_linker_rlib = false;
                while let Some(Ok(object_file)) = reader.next_entry() {
                    let name = std::str::from_utf8(object_file.header().identifier()).unwrap();
                    if name.ends_with(".rmeta") {
                        continue;
                    }

                    if object_file.header().size() == 0 {
                        continue;
                    }

                    // rlibs might contain dlls/sos/lib files which we don't want to include
                    //
                    // This catches .dylib, .so, .dll, .lib, .o, etc files that are not compatible with
                    // our "fat archive" linking process.
                    //
                    // We only trust `.rcgu.o` files to make it into the --all_load archive.
                    // This is a temporary stopgap to prevent issues with libraries that generate
                    // object files that are not compatible with --all_load.
                    // see https://github.com/DioxusLabs/dioxus/issues/4237
                    if !(name.ends_with(".rcgu.o") || name.ends_with(".obj")) {
                        keep_linker_rlib = true;
                        continue;
                    }

                    archive_has_contents = true;
                    out_ar
                        .append(&object_file.header().clone(), object_file)
                        .context("Failed to add object file to archive")?;
                }

                // Some rlibs contain weird artifacts that we don't want to include in the fat archive.
                // However, we still want them around in the linker in case the regular linker can handle them.
                if keep_linker_rlib {
                    compiler_rlibs.push(rlib.clone());
                }
            }

            let bytes = out_ar.into_inner().context("Failed to finalize archive")?;
            std::fs::write(&out_ar_path, bytes).context("Failed to write archive")?;
            log::debug!("Wrote fat archive to {out_ar_path:?}");

            // Run the ranlib command to index the archive. This slows down this process a bit,
            // but is necessary for some linkers to work properly.
            // We ignore its error in case it doesn't recognize the architecture
            if self.linker_flavor() == LinkerFlavor::Darwin
                && let Some(ranlib) = select_ranlib()
            {
                _ = tokio::process::Command::new(ranlib)
                    .arg(&out_ar_path)
                    .output()
                    .await;
            }
        }

        compiler_rlibs.dedup();

        // We're going to replace the first rlib in the args with our fat archive
        // And then remove the rest of the rlibs
        //
        // We also need to insert the -force_load flag to force the linker to load the archive
        let mut args = workspace_rustc_args.link_args.clone();

        if let Some(last_object) = args.iter().rposition(|arg| arg.ends_with(".o"))
            && archive_has_contents
        {
            match self.linker_flavor() {
                LinkerFlavor::WasmLld => {
                    args.insert(last_object, "--whole-archive".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.insert(last_object + 2, "--no-whole-archive".to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 3, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Gnu => {
                    args.insert(last_object, "-Wl,--whole-archive".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.insert(last_object + 2, "-Wl,--no-whole-archive".to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 3, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Darwin => {
                    args.insert(last_object, "-Wl,-force_load".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 2, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Msvc => {
                    args.insert(
                        last_object,
                        format!("/WHOLEARCHIVE:{}", out_ar_path.display()),
                    );
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 1, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Unsupported => {
                    log::error!("Unsupported platform for fat linking");
                }
            };
        }

        // Add custom args to the linkers
        match self.linker_flavor() {
            LinkerFlavor::Gnu => {
                // Export `main` so subsecond can use it for a reference point
                args.push("-Wl,--export-dynamic-symbol,main".to_string());
            }
            LinkerFlavor::Darwin => {
                args.push("-Wl,-exported_symbol,_main".to_string());
            }
            LinkerFlavor::Msvc => {
                // Prevent alsr from overflowing 32 bits
                args.push("/HIGHENTROPYVA:NO".to_string());

                // Export `main` so subsecond can use it for a reference point
                args.push("/EXPORT:main".to_string());
            }
            LinkerFlavor::WasmLld | LinkerFlavor::Unsupported => {}
        }

        // We also need to remove the `-o` flag since we want the linker output to end up in the
        // rust exe location, not in the deps dir as it normally would.
        if let Some(idx) = args
            .iter()
            .position(|arg| *arg == "-o" || *arg == "--output")
        {
            let _ = args.remove(idx + 1);
            let _ = args.remove(idx);
        }

        // same but windows support
        if let Some(idx) = args.iter().position(|arg| arg.starts_with("/OUT")) {
            let _ = args.remove(idx);
        }

        // We want to go through wasm-ld directly, so we need to remove the -flavor flag
        if let Some(flavor_idx) = args.iter().position(|arg| *arg == "-flavor") {
            let _ = args.remove(flavor_idx + 1);
            let _ = args.remove(flavor_idx);
        }

        // Set the output file
        match self.triple.operating_system {
            OperatingSystem::Windows => args.push(format!("/OUT:{}", exe.display())),
            _ => args.extend(["-o".to_string(), exe.display().to_string()]),
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;

        log::trace!("Fat linking with args: {:?} {:#?}", linker, args);
        log::trace!("Fat linking with env:");
        for e in rustc_args.envs.iter() {
            log::trace!("  {}={}", e.0, e.1);
        }

        // Handle windows command files
        let mut out_args = args.clone();
        if cfg!(windows) {
            let cmd_contents: String = out_args.iter().map(|f| format!("\"{f}\"")).join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display())];
        }

        // Add more search paths for the linker
        let mut command_envs: Vec<(String, String)> = rustc_args.envs.clone();

        // On linux, we need to set a more complete PATH for the linker to find its libraries
        if cfg!(target_os = "linux") {
            command_envs.push(("PATH".to_string(), std::env::var("PATH").unwrap()));
        }

        // Run the linker directly!
        let res = tokio::process::Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(command_envs.iter().map(|(k, v)| (k, v)))
            .output()
            .await?;

        if !res.status.success() {
            let mut combined = String::from_utf8_lossy(&res.stderr).into_owned();
            let out = String::from_utf8_lossy(&res.stdout);
            if !out.trim().is_empty() {
                combined = format!("{combined}\n{out}");
            }
            log::error!("Failed to generate fat binary:\n{}", combined.trim());
            return Err(anyhow::anyhow!("Failed to generate fat binary"));
        }

        if !res.stdout.is_empty() {
            let out = String::from_utf8_lossy(&res.stdout);
            log::trace!("Output from fat linking: {}", out.trim());
        }

        // Clean up the temps manually
        for f in args.iter().filter(|arg| arg.ends_with(".rcgu.o")) {
            _ = std::fs::remove_file(f);
        }

        // Cache the rlibs list
        _ = std::fs::write(
            &out_rlibs_list,
            compiler_rlibs
                .into_iter()
                .map(|s| s.display().to_string())
                .join("\n"),
        );

        Ok(())
    }

    fn linker_flavor(&self) -> LinkerFlavor {
        if let Some(custom) = self.custom_linker.as_ref() {
            let name = custom.file_name().unwrap().to_ascii_lowercase();
            match name.to_str() {
                Some("lld-link") => return LinkerFlavor::Msvc,
                Some("lld-link.exe") => return LinkerFlavor::Msvc,
                Some("wasm-ld") => return LinkerFlavor::WasmLld,
                Some("ld64.lld") => return LinkerFlavor::Darwin,
                Some("ld.lld") => return LinkerFlavor::Gnu,
                Some("ld.gold") => return LinkerFlavor::Gnu,
                Some("mold") => return LinkerFlavor::Gnu,
                Some("sold") => return LinkerFlavor::Gnu,
                Some("wild") => return LinkerFlavor::Gnu,
                _ => {}
            }
        }

        match self.triple.environment {
            target_lexicon::Environment::Gnu
            | target_lexicon::Environment::Gnuabi64
            | target_lexicon::Environment::Gnueabi
            | target_lexicon::Environment::Gnueabihf
            | target_lexicon::Environment::GnuLlvm => LinkerFlavor::Gnu,
            target_lexicon::Environment::Musl => LinkerFlavor::Gnu,
            target_lexicon::Environment::Android => LinkerFlavor::Gnu,
            target_lexicon::Environment::Msvc => LinkerFlavor::Msvc,
            target_lexicon::Environment::Macabi => LinkerFlavor::Darwin,
            _ => match self.triple.operating_system {
                OperatingSystem::Darwin(_) => LinkerFlavor::Darwin,
                OperatingSystem::IOS(_) => LinkerFlavor::Darwin,
                OperatingSystem::MacOSX(_) => LinkerFlavor::Darwin,
                OperatingSystem::Linux => LinkerFlavor::Gnu,
                OperatingSystem::Windows => LinkerFlavor::Msvc,
                _ => match self.triple.architecture {
                    target_lexicon::Architecture::Wasm32 => LinkerFlavor::WasmLld,
                    target_lexicon::Architecture::Wasm64 => LinkerFlavor::WasmLld,
                    _ => LinkerFlavor::Unsupported,
                },
            },
        }
    }

    fn frameworks_folder(&self) -> PathBuf {
        self.build_dir()
    }

    /// Select the linker to use for this platform.
    ///
    /// We prefer to use the rust-lld linker when we can since it's usually there.
    /// On macos, we use the system linker since macho files can be a bit finicky.
    ///
    /// This means we basically ignore the linker flavor that the user configured, which could
    /// cause issues with a custom linker setup. In theory, rust translates most flags to the right
    /// linker format.
    fn select_linker(&self) -> Result<PathBuf> {
        // Use a custom linker for non-crosscompile and crosscompile targets
        if matches!(
            self.triple.operating_system,
            OperatingSystem::Darwin(_) | OperatingSystem::Linux | OperatingSystem::Windows
        ) && let Ok(linker) = std::env::var("DX_HOST_LINKER")
        {
            return Ok(PathBuf::from(linker));
        }

        if let Ok(linker) = std::env::var("DX_LINKER") {
            return Ok(PathBuf::from(linker));
        }

        if let Some(linker) = self.custom_linker.clone() {
            return Ok(linker);
        }

        let cc = match self.linker_flavor() {
            LinkerFlavor::WasmLld => self.wasm_ld(),

            // On macOS, we use the system linker since it's usually there.
            // We could also use `lld` here, but it might not be installed by default.
            //
            // Note that this is *clang*, not `lld`.
            LinkerFlavor::Darwin => self.cc(),

            // On Linux, we use the system linker since it's usually there.
            LinkerFlavor::Gnu => self.cc(),

            // On windows, instead of trying to find the system linker, we just go with the lld.link
            // that rustup provides. It's faster and more stable then reyling on link.exe in path.
            LinkerFlavor::Msvc => self.lld_link(),

            // The rest of the platforms use `cc` as the linker which should be available in your path,
            // provided you have build-tools setup. On mac/linux this is the default, but on Windows
            // it requires msvc or gnu downloaded, which is a requirement to use rust anyways.
            //
            // The default linker might actually be slow though, so we could consider using lld or rust-lld
            // since those are shipping by default on linux as of 1.86. Window's linker is the really slow one.
            //
            // https://blog.rust-lang.org/2024/05/17/enabling-rust-lld-on-linux.html
            //
            // Note that "cc" is *not* a linker. It's a compiler! The arguments we pass need to be in
            // the form of `-Wl,<args>` for them to make it to the linker. This matches how rust does it
            // which is confusing.
            LinkerFlavor::Unsupported => self.cc(),
        };

        Ok(cc)
    }

    /// Return the path to the `cc` compiler
    ///
    /// This is used for the patching system to run the linker.
    /// We could also just use lld given to us by rust itself.
    pub fn cc(&self) -> PathBuf {
        PathBuf::from("cc")
    }

    /// The windows linker
    pub fn lld_link(&self) -> PathBuf {
        self.gcc_ld_dir().join("lld-link")
    }

    pub fn wasm_ld(&self) -> PathBuf {
        self.gcc_ld_dir().join("wasm-ld")
    }

    fn gcc_ld_dir(&self) -> PathBuf {
        self.sysroot
            .join("lib")
            .join("rustlib")
            .join(Triple::host().to_string())
            .join("bin")
            .join("gcc-ld")
    }

    fn build_command(&self, mode: &BuildMode) -> Result<tokio::process::Command> {
        match mode {
            // We're assembling rustc directly, so we need to be *very* careful. Cargo sets rustc's
            // env up very particularly, and we want to match it 1:1 but with some changes.
            //
            // To do this, we reset the env completely, and then pass every env var that the original
            // rustc process had 1:1.
            //
            // We need to unset a few things, like the RUSTC wrappers and then our special env var
            // indicating that dx itself is the compiler. If we forget to do this, then the compiler
            // ends up doing some recursive nonsense and dx is trying to link instead of compiling.
            //
            // todo: maybe rustc needs to be found on the FS instead of using the one in the path?
            BuildMode::Thin {
                workspace_rustc, ..
            } => {
                let tip_crate_key = format!("{}.bin", self.tip_crate_name());
                let rustc_args = workspace_rustc
                    .rustc_args
                    .get(&tip_crate_key)
                    .with_context(|| {
                        format!(
                            "Missing captured rustc args for the tip crate '{tip_crate_key}' \
                             (available: {:?})",
                            workspace_rustc.rustc_args.keys().collect::<Vec<_>>()
                        )
                    })?;

                let mut cmd = tokio::process::Command::new("rustc");

                // Replay the build in the working directory captured by the rustc wrapper when
                // the args were recorded, falling back to the workspace dir for old captures
                // that predate the field (serde(default) gives an empty path).
                let cwd = if rustc_args.cwd.as_os_str().is_empty() {
                    &self.workspace_dir
                } else {
                    &rustc_args.cwd
                };

                let _ = cmd
                    .current_dir(cwd)
                    .env_clear()
                    .args(rustc_args.args[1..].iter())
                    .env_remove("RUSTC_WORKSPACE_WRAPPER")
                    .env_remove("RUSTC_WRAPPER")
                    .env_remove(rustc::DX_RUSTC_WRAPPER_ENV_VAR)
                    .envs(self.cargo_build_env_vars(mode)?)
                    .arg(format!("-Clinker={}", path_to_me()?.display()));

                if is_wasm_or_wasi(&self.triple) {
                    let _ = cmd.arg("-Crelocation-model=pic");
                }

                log::debug!("Direct rustc: {cmd:#?}");

                let _ = cmd.envs(rustc_args.envs.iter().cloned());

                // tracing::trace!("Setting env vars: {:#?}", rustc_args.envs);

                Ok(cmd)
            }

            // For Base and Fat builds, we use a regular cargo setup, but we might need to intercept
            // rustc itself in case we're hot-patching and need a reliable rustc environment to
            // continuously recompile the workspace with.
            //
            // RUSTC_WORKSPACE_WRAPPER routes only the rustc invocations of *workspace member*
            // crates through us - exactly the crates a hotpatch can replay. This avoids
            // capturing every crates.io dependency as well, and (unlike RUSTC_WRAPPER) it is
            // part of cargo's unit hash: artifacts built by a plain `cargo build` are
            // considered stale by our next build, so they get recompiled - and therefore
            // re-captured - automatically.
            //
            // We've also had a number of issues with incorrect canonicalization when passing paths
            // through envs on windows, hence the frequent use of dunce::canonicalize.
            _ => {
                let mut cmd = tokio::process::Command::new("cargo");

                let _ = cmd
                    .arg("rustc")
                    .current_dir(&self.crate_dir)
                    .arg("--message-format")
                    .arg("json-diagnostic-rendered-ansi")
                    .arg("--color")
                    .arg("always")
                    .args(self.cargo_build_arguments(mode))
                    .envs(self.cargo_build_env_vars(mode)?);

                if mode == &BuildMode::Fat {
                    let args_dir = self.rustc_wrapper_args_dir();
                    let _ = std::fs::create_dir_all(&args_dir);

                    let _ = cmd
                        .env(
                            rustc::DX_RUSTC_WRAPPER_ENV_VAR,
                            dunce::canonicalize(&args_dir)
                                .with_context(|| {
                                    format!("Failed to canonicalize {}", args_dir.display())
                                })?
                                .display()
                                .to_string(),
                        )
                        .env(
                            "RUSTC_WORKSPACE_WRAPPER",
                            path_to_me()?.display().to_string(),
                        );
                }

                log::debug!("Cargo: {cmd:#?}");

                Ok(cmd)
            }
        }
    }

    /// Create a list of arguments for cargo builds
    ///
    /// We always use `cargo rustc` *or* `rustc` directly. This means we can pass extra flags like
    /// `-C` arguments directly to the compiler.
    #[allow(clippy::vec_init_then_push)]
    fn cargo_build_arguments(&self, mode: &BuildMode) -> Vec<String> {
        let mut cargo_args = Vec::with_capacity(4);

        // Add required profile flags. --release overrides any custom profiles.
        cargo_args.push("--profile".to_string());
        cargo_args.push(self.profile.to_string());

        // Pass the appropriate target to cargo. We *always* specify a target which is somewhat helpful for preventing thrashing
        cargo_args.push("--target".to_string());
        cargo_args.push(self.triple.to_string());

        // We always run in verbose since the CLI itself is the one doing the presentation
        if self.verbose > 0 {
            cargo_args.push(format!("-{}", "v".repeat(usize::from(self.verbose))));
        }

        if self.no_default_features {
            cargo_args.push("--no-default-features".to_string());
        }

        if !self.features.is_empty() {
            cargo_args.push("--features".to_string());
            cargo_args.push(self.features.join(" "));
        }

        // We *always* set the package since that's discovered from cargo metadata
        cargo_args.push(String::from("-p"));
        cargo_args.push(self.package.clone());

        // Set the executable
        match self.crate_target.kind() {
            TargetKind::Bin => cargo_args.push("--bin".to_string()),
            TargetKind::Lib(_) => cargo_args.push("--lib".to_string()),
            TargetKind::ExampleBin => cargo_args.push("--example".to_string()),
            _ => {}
        };
        cargo_args.push(self.executable_name().to_string());

        // Merge in extra args. Order shouldn't really matter.
        cargo_args.extend(self.extra_cargo_args.clone());
        cargo_args.push("--".to_string());
        cargo_args.extend(self.extra_rustc_args.clone());

        if cfg!(target_os = "windows")
            && !self
                .extra_rustc_args
                .iter()
                .any(|f| f.starts_with("-Clink-arg=/SUBSYSTEM:"))
        {
            // On windows, we pass /SUBSYSTEM:WINDOWS to prevent a console from appearing
            cargo_args.push("-Clink-arg=/SUBSYSTEM:WINDOWS".to_string());
            // We also need to set the entry point to mainCRTStartup to avoid windows looking
            // for a WinMain function
            cargo_args.push("-Clink-arg=/ENTRY:mainCRTStartup".to_string());
        }

        // TODO
        // The bundle splitter needs relocation data to create a call-graph.
        // This will automatically be erased by wasm-opt during the optimization step.
        // if self.platform == Platform::Web && self.wasm_split {
        //     cargo_args.push("-Clink-args=--emit-relocs".to_string());
        // }

        // dx *always* links android and thin builds
        if self.custom_linker.is_some() || matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            cargo_args.push(format!(
                "-Clinker={}",
                path_to_me().expect("can't find dx").display()
            ));
        }

        // TODO
        // for debuggability, we need to make sure android studio can properly understand our build
        // https://stackoverflow.com/questions/68481401/debugging-a-prebuilt-shared-library-in-android-studio
        // if self.platform == Platform::Android {
        //     cargo_args.push("-Clink-arg=-Wl,--build-id=sha1".to_string());
        // }

        // Handle frameworks/dylibs by setting the rpath
        // This is dependent on the bundle structure - in this case, appimage and appbundle for mac/linux
        // todo: we need to figure out what to do for windows
        match self.triple.operating_system {
            OperatingSystem::Darwin(_) | OperatingSystem::IOS(_) => {
                cargo_args.push("-Clink-arg=-Wl,-rpath,@executable_path/../Frameworks".to_string());
                cargo_args.push("-Clink-arg=-Wl,-rpath,@executable_path".to_string());
            }
            OperatingSystem::Linux => {
                cargo_args.push("-Clink-arg=-Wl,-rpath,$ORIGIN/../lib".to_string());
                cargo_args.push("-Clink-arg=-Wl,-rpath,$ORIGIN".to_string());
            }
            _ => {}
        }

        // Our fancy hot-patching engine needs a lot of customization to work properly.
        //
        // These args are mostly intended to be passed when *fat* linking but are generally fine to
        // pass for both fat and thin linking.
        //
        // We need save-temps and no-dead-strip in both cases though. When we run `cargo rustc` with
        // these args, they will be captured and re-ran for the fast compiles in the future, so whatever
        // we set here will be set for all future hot patches too.
        if matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            // rustc gives us some portable flags required:
            // - link-dead-code: prevents rust from passing -dead_strip to the linker since that's the default.
            // - save-temps=true: keeps the incremental object files around, which we need for manually linking.
            cargo_args.extend_from_slice(&[
                "-Csave-temps=true".to_string(),
                "-Clink-dead-code".to_string(),
            ]);

            // TODO
            // We need to set some extra args that ensure all symbols make it into the final output
            // and that the linker doesn't strip them out.
            //
            // This basically amounts of -all_load or --whole-archive, depending on the linker.
            // We just assume an ld-like interface on macos and a gnu-ld interface elsewhere.
            //
            // macOS/iOS use ld64 but through the `cc` interface.
            // cargo_args.push("-Clink-args=-Wl,-all_load".to_string());
            //
            // Linux and Android fit under this umbrella, both with the same clang-like entrypoint
            // and the gnu-ld interface.
            //
            // cargo_args.push("-Clink-args=-Wl,--whole-archive".to_string());
            //
            // If windows -Wl,--whole-archive is required since it follows gnu-ld convention.
            // There might be other flags on windows - we haven't tested windows thoroughly.
            //
            // cargo_args.push("-Clink-args=-Wl,--whole-archive".to_string());
            // https://learn.microsoft.com/en-us/cpp/build/reference/wholearchive-include-all-library-object-files?view=msvc-170
            //
            // ------------------------------------------------------------
            //
            // if web, -Wl,--whole-archive is required since it follows gnu-ld convention.
            //
            // We also use --no-gc-sections and --export-table and --export-memory  to push
            // said symbols into the export table.
            //
            // We use --emit-relocs to build up a solid call graph.
            //
            // rust uses its own wasm-ld linker which can be found here (it's just gcc-ld with a `-target wasm` flag):
            // - ~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld
            // - ~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld/wasm-ld
            //
            // Note that we can't use --export-all, unfortunately, since some symbols are internal
            // to wasm-bindgen and exporting them causes the JS generation to fail.
            //
            // We are basically replicating what emscripten does here with its dynamic linking
            // approach where the MAIN_MODULE is very "fat" and exports the necessary arguments
            // for the side modules to be linked in. This guide is really helpful:
            //
            // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md
            //
            // The tricky one is -Ctarget-cpu=mvp, which prevents rustc from generating externref
            // entries.
            //
            // https://blog.rust-lang.org/2024/09/24/webassembly-targets-change-in-default-target-features/#disabling-on-by-default-webassembly-proposals
            //
            // It's fine that these exist in the base module but not in the patch.
            if is_wasm_or_wasi(&self.triple) {
                // cargo_args.push("-Ctarget-cpu=mvp".into()); // disabled due to changes in wasm-bindgen
                cargo_args.push("-Clink-arg=--no-gc-sections".into());
                cargo_args.push("-Clink-arg=--growable-table".into());
                cargo_args.push("-Clink-arg=--export-table".into());
                cargo_args.push("-Clink-arg=--export-memory".into());
                cargo_args.push("-Clink-arg=--emit-relocs".into());
                cargo_args.push("-Clink-arg=--export=__stack_pointer".into());
                cargo_args.push("-Clink-arg=--export=__heap_base".into());
                cargo_args.push("-Clink-arg=--export=__data_end".into());
            }
        }

        cargo_args
    }

    fn cargo_build_env_vars(&self, mode: &BuildMode) -> Result<Vec<(&'static str, String)>> {
        let mut env_vars = vec![];

        // TODO
        // Make sure to set all the crazy android flags. Cross-compiling is hard, man.
        // if self.platform == Platform::Android {
        //     env_vars.extend(self.android_env_vars()?);
        // };

        // If we're either zero-linking or using a custom linker, make `dx` itself do the linking.
        if self.custom_linker.is_some() || matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            link::LinkAction {
                triple: self.triple.clone(),
                linker: self.custom_linker.clone(),
                link_err_file: dunce::canonicalize(self.link_err_file())?,
                link_args_file: dunce::canonicalize(self.link_args_file())?,
            }
            .write_env_vars(&mut env_vars)?;
        }

        // TODO
        // Disable reference types on wasm when using hotpatching
        // https://blog.rust-lang.org/2024/09/24/webassembly-targets-change-in-default-target-features/#disabling-on-by-default-webassembly-proposals
        // if self.platform == Platform::Web
        //     && matches!(ctx.mode, BuildMode::Thin { .. } | BuildMode::Fat)
        // {
        //     env_vars.push(("RUSTFLAGS", {
        //         let mut rust_flags = std::env::var("RUSTFLAGS").unwrap_or_default();
        //         rust_flags.push_str(" -Ctarget-cpu=mvp");
        //         rust_flags
        //     }));
        // }

        Ok(env_vars)
    }

    fn link_args_file(&self) -> PathBuf {
        self.exe_dir().join("link_args.txt")
    }

    fn link_err_file(&self) -> PathBuf {
        self.exe_dir().join("link_err.txt")
    }

    fn rustc_wrapper_args_dir(&self) -> PathBuf {
        self.exe_dir().join("rustc_wrapper_args")
    }

    /// The rustc crate name (hyphens → underscores) of the tip target, as used by
    /// `--crate-name` and the per-crate capture keys.
    fn tip_crate_name(&self) -> String {
        self.crate_target.name().replace('-', "_")
    }

    /// The workspace package name of the tip crate (hyphens → underscores).
    ///
    /// This can differ from `tip_crate_name()` when the binary target is named differently than
    /// its package (e.g. package `browser` with `[[bin]] name = "blitz"`). Use this for
    /// workspace-graph lookups (which are keyed by package name) and `tip_crate_name()` for
    /// rustc `--crate-name` keys like `{name}.bin`.
    fn tip_package_name(&self) -> String {
        self.package.replace('-', "_")
    }

    /// The source directories of every workspace member.
    fn workspace_member_dirs(&self) -> Vec<PathBuf> {
        let member_ids: HashSet<&cargo_metadata::PackageId> =
            self.metadata.workspace_members.iter().collect();

        self.metadata
            .packages
            .iter()
            .filter(|package| member_ids.contains(&package.id))
            .filter_map(|package| package.manifest_path.parent().map(PathBuf::from))
            .collect()
    }

    /// Map a changed file path to the workspace crate it belongs to.
    ///
    /// Returns the crate name in rustc convention (hyphens → underscores), matching the
    /// `--crate-name` arg used by rustc and the keys in `workspace_rustc_args.rustc_args`.
    ///
    /// Finds the workspace member whose crate directory is the longest prefix of the file path.
    fn file_to_workspace_crate(&self, file: &Path) -> Option<String> {
        let member_ids: HashSet<&cargo_metadata::PackageId> =
            self.metadata.workspace_members.iter().collect();

        let mut best_match: Option<(String, usize)> = None;

        for package in self
            .metadata
            .packages
            .iter()
            .filter(|package| member_ids.contains(&package.id))
        {
            let Some(crate_dir) = package.manifest_path.parent() else {
                continue;
            };

            if let Ok(relative) = file.strip_prefix(crate_dir) {
                let depth = relative.components().count();
                let is_better = best_match
                    .as_ref()
                    .is_none_or(|(_, best_depth)| depth < *best_depth);

                if is_better {
                    best_match = Some((package.name.replace('-', "_"), depth));
                }
            }
        }

        best_match.map(|(name, _)| name)
    }

    /// The transitive workspace dependencies of a crate (BFS over the resolve graph,
    /// following edges forward), including the crate itself.
    fn workspace_dep_closure(&self, crate_name: &str) -> BTreeSet<String> {
        let Some(resolve) = self.metadata.resolve.as_ref() else {
            return BTreeSet::new();
        };

        let member_ids: HashSet<&cargo_metadata::PackageId> =
            self.metadata.workspace_members.iter().collect();
        let crate_name = crate_name.replace('-', "_");

        let mut names: BTreeSet<String> = BTreeSet::new();
        let mut visited: HashSet<&cargo_metadata::PackageId> = HashSet::new();
        let mut queue: Vec<&cargo_metadata::PackageId> = resolve
            .nodes
            .iter()
            .filter(|node| {
                member_ids.contains(&node.id)
                    && self
                        .metadata
                        .packages
                        .iter()
                        .find(|package| package.id == node.id)
                        .is_some_and(|package| package.name.replace('-', "_") == crate_name)
            })
            .map(|node| &node.id)
            .collect();

        while let Some(package_id) = queue.pop() {
            if !visited.insert(package_id) {
                continue;
            }

            if member_ids.contains(package_id)
                && let Some(package) = self
                    .metadata
                    .packages
                    .iter()
                    .find(|package| package.id == *package_id)
            {
                let _ = names.insert(package.name.replace('-', "_"));
            }

            for node in resolve.nodes.iter().filter(|node| node.id == *package_id) {
                queue.extend(node.dependencies.iter());
            }
        }

        names
    }

    /// Ensure every workspace member in the tip's dependency tree has a captured rustc
    /// invocation in the args dir. Members missing a capture (e.g. built by an older
    /// `cargo-hot` or by plain `cargo build`) are cleaned so the fat build recompiles -
    /// and therefore captures - them.
    async fn ensure_workspace_captures(&self) -> Result<()> {
        let dep_closure = self.workspace_dep_closure(&self.tip_package_name());

        let captured: HashSet<String> = std::fs::read_dir(self.rustc_wrapper_args_dir())
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|entry| {
                        entry.file_name().to_str().and_then(|name| {
                            name.strip_suffix(".lib.json")
                                .or_else(|| name.strip_suffix(".bin.json"))
                                .map(str::to_string)
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        for member in dep_closure
            .iter()
            .filter(|member| !captured.contains(member.as_str()))
        {
            log::debug!(
                "No rustc capture for workspace member '{member}'; cleaning so the fat \
                 build recompiles and captures it"
            );

            let status = process::Command::new("cargo")
                .arg("clean")
                .arg("-p")
                .arg(member.replace('_', "-"))
                .current_dir(&self.workspace_dir)
                .status()
                .await
                .with_context(|| format!("Failed to run `cargo clean -p {member}`"))?;

            ensure!(
                status.success(),
                "Failed to clean workspace member '{member}' before the fat build"
            );
        }

        Ok(())
    }

    /// The direct workspace dependents of a crate - the workspace members that depend on it.
    fn workspace_dependents_of(&self, crate_name: &str) -> Vec<String> {
        let Some(resolve) = self.metadata.resolve.as_ref() else {
            return Vec::new();
        };

        let member_ids: HashSet<&cargo_metadata::PackageId> =
            self.metadata.workspace_members.iter().collect();
        let crate_name = crate_name.replace('-', "_");

        // The ids of every node that is a workspace member with the target name
        let target_ids: HashSet<&cargo_metadata::PackageId> = resolve
            .nodes
            .iter()
            .filter(|node| {
                member_ids.contains(&node.id)
                    && self
                        .metadata
                        .packages
                        .iter()
                        .find(|package| package.id == node.id)
                        .is_some_and(|package| package.name.replace('-', "_") == crate_name)
            })
            .map(|node| &node.id)
            .collect();

        if target_ids.is_empty() {
            return Vec::new();
        }

        let mut dependents = Vec::new();
        let mut seen = HashSet::new();

        for node in resolve.nodes.iter() {
            if member_ids.contains(&node.id)
                && node.dependencies.iter().any(|dep| target_ids.contains(dep))
                && let Some(package) = self
                    .metadata
                    .packages
                    .iter()
                    .find(|package| package.id == node.id)
                && seen.insert(package.name.replace('-', "_"))
            {
                dependents.push(package.name.replace('-', "_"));
            }
        }

        dependents
    }

    /// The transitive workspace dependents of a crate (BFS over `workspace_dependents_of`),
    /// including the crate itself.
    fn workspace_dependent_closure(&self, crate_name: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut queue = vec![crate_name.to_string()];

        while let Some(current) = queue.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }

            for dependent in self.workspace_dependents_of(&current) {
                if !seen.contains(&dependent) {
                    queue.push(dependent);
                }
            }
        }

        seen
    }

    /// Order the crates to replay for a hotpatch so dependencies compile before their dependents.
    ///
    /// The tip crate is excluded - it's always recompiled separately through cargo.
    fn workspace_hotpatch_replay_order(
        &self,
        modified_crates: &HashSet<String>,
    ) -> Result<Vec<String>> {
        let tip = self.tip_package_name();

        // The crates to replay, excluding the tip
        let crates: BTreeSet<String> = modified_crates
            .iter()
            .filter(|crate_name| **crate_name != tip)
            .cloned()
            .collect();

        // In-degree = how many of this crate's workspace deps are also being replayed
        let mut in_degree: HashMap<String, usize> = crates
            .iter()
            .map(|crate_name| (crate_name.clone(), 0))
            .collect();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for crate_name in crates.iter() {
            for dependent in self.workspace_dependents_of(crate_name) {
                if crates.contains(&dependent) {
                    *in_degree.get_mut(&dependent).unwrap() += 1;
                    dependents
                        .entry(crate_name.clone())
                        .or_default()
                        .push(dependent.clone());
                }
            }
        }

        // Kahn's algorithm with a BTreeSet queue so ties break lexicographically
        let mut queue: BTreeSet<String> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(crate_name, _)| crate_name.clone())
            .collect();

        let mut order = Vec::new();

        while let Some(current) = queue.iter().next().cloned() {
            let _ = queue.remove(&current);
            order.push(current.clone());

            if let Some(dependent_list) = dependents.get(&current) {
                for dependent in dependent_list {
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            let _ = queue.insert(dependent.clone());
                        }
                    }
                }
            }
        }

        if order.len() != crates.len() {
            let unresolved: Vec<String> = in_degree
                .iter()
                .filter(|&(_, &degree)| degree > 0)
                .map(|(crate_name, _)| crate_name.clone())
                .collect();

            return Err(anyhow!(
                "Dependency cycle detected among workspace crates: {unresolved:?}"
            ));
        }

        Ok(order)
    }

    /// Get the rustc args for replaying a workspace dependency crate, preferring the lib target.
    fn workspace_hotpatch_replay_args(
        &self,
        workspace_rustc_args: &rustc::WorkspaceRustcArgs,
        crate_name: &str,
    ) -> Option<rustc::Args> {
        let lib_key = format!("{crate_name}.lib");
        if let Some(args) = workspace_rustc_args.rustc_args.get(&lib_key) {
            return Some(args.clone());
        }

        let bin_key = format!("{crate_name}.bin");
        workspace_rustc_args.rustc_args.get(&bin_key).cloned()
    }

    /// Resolve the on-disk rlibs for every replayed workspace crate, preserving the link order
    /// from the original fat build.
    fn workspace_hotpatch_link_rlibs(
        &self,
        workspace_rustc_args: &rustc::WorkspaceRustcArgs,
        replayed_crates: &[String],
    ) -> Result<Vec<PathBuf>> {
        let mut wanted = HashSet::new();

        for crate_name in replayed_crates {
            let Some(rustc_args) = workspace_rustc_args
                .rustc_args
                .get(&format!("{crate_name}.lib"))
            else {
                // No capture means the crate wasn't rebuilt (or never built at all) - its code
                // is still what the fat binary has, so the stubs resolve it as usual.
                log::debug!(
                    "Skipping rlib for workspace crate '{crate_name}': no captured rustc args"
                );
                continue;
            };

            let rlib = self
                .find_rlib_for_crate(crate_name, rustc_args)
                .with_context(|| {
                    format!("Could not find rlib for workspace crate '{crate_name}'")
                })?;

            let _ = wanted.insert(rlib);
        }

        // Preserve the link order from the original fat build for any rlibs that appear
        // in the captured link args.
        let mut ordered = Vec::new();
        let mut seen = HashSet::new();

        for arg in &workspace_rustc_args.link_args {
            if !arg.ends_with(".rlib") {
                continue;
            }

            let path = PathBuf::from(arg);
            if wanted.contains(&path) && seen.insert(path.clone()) {
                ordered.push(path);
            }
        }

        // Any rlibs not in the captured link order get appended at the end.
        let mut remaining: Vec<_> = wanted.into_iter().filter(|p| !seen.contains(p)).collect();
        remaining.sort();
        ordered.extend(remaining);

        Ok(ordered)
    }

    /// Locate the rlib rustc produced for `crate_name` using its captured args.
    fn find_rlib_for_crate(&self, crate_name: &str, rustc_args: &rustc::Args) -> Result<PathBuf> {
        // Extract --out-dir from the captured args
        let out_dir = rustc_args
            .args
            .iter()
            .zip(rustc_args.args.iter().skip(1))
            .find(|(flag, _)| *flag == "--out-dir")
            .map(|(_, dir)| PathBuf::from(dir))
            .with_context(|| format!("No --out-dir in captured rustc args for '{crate_name}'"))?;

        // Extract -C extra-filename from captured args.
        // Cargo passes this to rustc to disambiguate output filenames via metadata hash.
        // Handle all forms: `-Cextra-filename=X`, `-C extra-filename=X`, and `-C` `extra-filename=X`.
        let extra_filename = rustc_args.args.iter().enumerate().find_map(|(i, arg)| {
            arg.strip_prefix("-Cextra-filename=")
                .map(|s| s.to_string())
                .or_else(|| {
                    if arg == "-C" {
                        rustc_args.args.get(i + 1).and_then(|next| {
                            next.strip_prefix("extra-filename=").map(|s| s.to_string())
                        })
                    } else {
                        None
                    }
                })
        });

        // If we have an exact extra-filename, construct the precise rlib path.
        if let Some(extra) = &extra_filename {
            let exact = out_dir.join(format!("lib{crate_name}{extra}.rlib"));
            if exact.exists() {
                return Ok(exact);
            }
        }

        // Fallback: glob for lib<crate_name>-<hash>.rlib in the output directory.
        // Prefer the most recently modified rlib to avoid picking up stale artifacts.
        let prefix = format!("lib{crate_name}-");
        let mut best: Option<(PathBuf, SystemTime)> = None;

        for entry in std::fs::read_dir(&out_dir)
            .with_context(|| format!("Could not read --out-dir '{}'", out_dir.display()))?
            .flatten()
        {
            if let Some(name) = entry.file_name().to_str()
                && name.starts_with(&prefix)
                && name.ends_with(".rlib")
                && let Ok(meta) = entry.metadata()
                && let Ok(mtime) = meta.modified()
                && best.as_ref().is_none_or(|(_, t)| mtime > *t)
            {
                best = Some((entry.path(), mtime));
            }
        }

        best.map(|(path, _)| path).with_context(|| {
            format!(
                "Could not find rlib for '{crate_name}' in {}",
                out_dir.display()
            )
        })
    }

    /// Recompile a workspace dependency crate using the rustc args captured during the fat build.
    ///
    /// The captured args are replayed verbatim (except for the linker override) so the rlib is
    /// rewritten in place with the latest code, ready to be linked into the patch.
    async fn compile_dep_crate(&self, crate_name: &str, rustc_args: &rustc::Args) -> Result<()> {
        use tokio::io::AsyncBufReadExt;

        let mut cmd = tokio::process::Command::new("rustc");

        // Drop the linker override - the replay runs the compiler directly and we don't want
        // to intercept (or loop through) the linker for dependency crates.
        let args = rustc_args.args[1..]
            .iter()
            .filter(|arg| !arg.as_str().starts_with("-Clinker=") && arg.as_str() != "-Clinker")
            .cloned()
            .collect::<Vec<_>>();

        let _ = cmd.args(&args);

        // Replay in the captured working directory so relative paths resolve the same way
        let cwd = if rustc_args.cwd.as_os_str().is_empty() {
            &self.workspace_dir
        } else {
            &rustc_args.cwd
        };
        let _ = cmd.current_dir(cwd);

        // Drop the wrapper/linker interception env vars - the replay is a plain rustc invocation
        let mut envs = rustc_args.envs.clone();
        envs.retain_mut(|(key, _)| {
            !key.starts_with("DX_LINK")
                && !matches!(
                    key.as_str(),
                    "RUSTC_WORKSPACE_WRAPPER"
                        | "RUSTC_WRAPPER"
                        | "DX_RUSTC"
                        | "CARGO_MAKEFLAGS"
                        | "MAKEFLAGS"
                )
        });
        let _ = cmd.env_clear().envs(envs);

        // Wasm needs a different relocation model for thin linking
        if is_wasm_or_wasi(&self.triple) {
            let _ = cmd.arg("-Crelocation-model=pic");
        }

        log::debug!("Replaying rustc for workspace crate '{crate_name}'");

        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("Failed to spawn rustc for dep crate")?;

        let stdout = tokio::io::BufReader::new(child.stdout.take().unwrap());
        let stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());

        let mut stdout = stdout.lines();
        let mut stderr = stderr.lines();
        let mut rendered_diagnostics: Vec<String> = Vec::new();

        loop {
            use cargo_metadata::Message;
            use cargo_metadata::diagnostic::Diagnostic;

            let line = tokio::select! {
                Ok(Some(line)) = stdout.next_line() => line,
                Ok(Some(line)) = stderr.next_line() => line,
                else => break,
            };

            let Some(Ok(message)) = Message::parse_stream(std::io::Cursor::new(line)).next() else {
                continue;
            };

            match message {
                Message::CompilerMessage(msg) => {
                    println!("{}", msg.message);
                }
                Message::TextLine(line) => {
                    // Direct rustc diagnostics are emitted as JSON on stderr
                    if let Ok(diag) = serde_json::from_str::<Diagnostic>(&line)
                        && let Some(rendered) = diag.rendered
                    {
                        println!("{rendered}");
                        rendered_diagnostics.push(rendered);
                    }
                }
                _ => {}
            }
        }

        let status = child.wait().await?;

        if !status.success() {
            let diagnostics = rendered_diagnostics.join("\n");
            return Err(anyhow!(
                "Replay of workspace crate '{crate_name}' failed{}",
                if diagnostics.is_empty() {
                    String::new()
                } else {
                    format!(":\n{diagnostics}")
                }
            ));
        }

        Ok(())
    }

    /// Fold a crate's dep-info file into the filemap, mapping every input file to the crate.
    ///
    /// Existing entries are left alone unless `overwrite` is set - dependency crates are
    /// registered after the tip so that a file shared between them triggers the dependency's
    /// replay (whose closure then also recompiles the tip).
    fn refresh_dep_info(&self, crate_name: &str, rustc_args: &rustc::Args, overwrite: bool) {
        let Some(dep_info_path) = rustc::dep_info_path_for_rustc_args(&rustc_args.args) else {
            return;
        };

        // Rustc writes the dep-info relative to the compilation's working directory
        let cwd = if rustc_args.cwd.as_os_str().is_empty() {
            &self.workspace_dir
        } else {
            &rustc_args.cwd
        };

        let files = rustc::parse_dep_info_files(&dep_info_path, cwd);
        if files.is_empty() {
            log::debug!(
                "No dep-info files found for '{crate_name}' at {}",
                dep_info_path.display()
            );
            return;
        }

        let mut depinfo = self.depinfo.lock().unwrap();
        for file in files {
            if overwrite || !depinfo.contains_key(&file) {
                let _ = depinfo.insert(file, crate_name.to_string());
            }
        }
    }

    /// Load the rustc args the wrapper captured for every crate during the build, along with the
    /// linker args of the tip crate's final link invocation.
    fn load_rustc_argset(&self) -> Result<rustc::WorkspaceRustcArgs> {
        let link_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let mut workspace_rustc_args = rustc::WorkspaceRustcArgs::new(link_args);

        if let Ok(entries) = std::fs::read_dir(self.rustc_wrapper_args_dir()) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "json")
                    && let Ok(contents) = std::fs::read_to_string(&path)
                    && let Ok(args) = serde_json::from_str::<rustc::Args>(&contents)
                    && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
                {
                    let _ = workspace_rustc_args
                        .rustc_args
                        .insert(stem.to_string(), args);
                }
            }
        }

        Ok(workspace_rustc_args)
    }

    fn windows_command_file(&self) -> PathBuf {
        self.exe_dir().join("windows_command.txt")
    }
}

fn select_ranlib() -> Option<PathBuf> {
    // prefer the modern llvm-ranlib if they have it
    which::which("llvm-ranlib")
        .or_else(|_| which::which("ranlib"))
        .ok()
}

fn path_to_me() -> Result<PathBuf> {
    dunce::canonicalize(std::env::current_exe().context("Failed to find cargo-hot")?)
        .context("Failed to find cargo-hot")
}

async fn read_batch<T>(
    receiver: &mut mpsc::Receiver<T>,
    buffer: &mut Vec<T>,
    duration: Duration,
) -> usize {
    use tokio::time;

    let Some(item) = receiver.recv().await else {
        return 0;
    };

    buffer.push(item);

    let mut n = 1;

    loop {
        let Ok(Some(item)) = time::timeout(duration, receiver.recv()).await else {
            break;
        };

        buffer.push(item);
        n += 1;
    }

    n
}

/// Whether the target triple is a wasm32/wasm64 or wasi target.
fn is_wasm_or_wasi(triple: &Triple) -> bool {
    matches!(
        triple.architecture,
        target_lexicon::Architecture::Wasm32 | target_lexicon::Architecture::Wasm64
    ) || triple.operating_system == target_lexicon::OperatingSystem::Wasi
}

/// Detects if `dx` is being ran in a WSL environment.
///
/// We determine this based on whether the keyword `microsoft` or `wsl` is contained within the `WSL_1` or `WSL_2` files.
/// This may fail in the future as it isn't guaranteed by Microsoft.
/// See <https://github.com/microsoft/WSL/issues/423#issuecomment-221627364>
fn is_wsl() -> bool {
    const WSL_1: &str = "/proc/sys/kernel/osrelease";
    const WSL_2: &str = "/proc/version";
    const WSL_KEYWORDS: [&str; 2] = ["microsoft", "wsl"];

    // Test 1st File
    if let Ok(content) = std::fs::read_to_string(WSL_1) {
        let lowercase = content.to_lowercase();
        for keyword in WSL_KEYWORDS {
            if lowercase.contains(keyword) {
                return true;
            }
        }
    }

    // Test 2nd File
    if let Ok(content) = std::fs::read_to_string(WSL_2) {
        let lowercase = content.to_lowercase();
        for keyword in WSL_KEYWORDS {
            if lowercase.contains(keyword) {
                return true;
            }
        }
    }

    false
}
