use cargo_hot::Result;
use cargo_hot::hotpatch;
use cargo_hot::link::{self, LinkerFlavor};
use cargo_hot::rustc;
use cargo_hot_protocol::server;

use cargo::GlobalContext;
use cargo::core::{Target, TargetKind, Workspace};
use cargo::util::{Filesystem, command_prelude::*};

use anyhow::{Context, anyhow};
use itertools::Itertools;
use serde::Deserialize;
use target_lexicon::{Architecture, OperatingSystem, Triple};
use tokio::process;
use tokio::sync::mpsc;

use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[tokio::main]
async fn main() -> Result<ExitCode> {
    #[cfg(debug_assertions)]
    tracing_subscriber::fmt::init();

    if rustc::is_wrapping_rustc() {
        rustc::run_rustc().await;
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

    let ws = args.workspace(&gctx)?;
    let server = Server::new(&gctx, ws, &args).await?;

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
        .after_help(color_print::cstr!(
            "Run `<cyan,bold>cargo help run</>` for more detailed information.\n"
        ))
}

#[derive(Debug)]
pub struct Server {
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
    no_default_features: bool,
    target_dir: Filesystem,
    custom_linker: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Build {
    exe: PathBuf,
    is_fresh: bool,
    direct_rustc: rustc::Args,
    time_start: SystemTime,
    patch_cache: Option<Arc<hotpatch::Cache>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuildMode {
    Fat,
    Thin {
        rustc_args: rustc::Args,
        changed_files: Vec<PathBuf>,
        aslr_reference: u64,
        cache: Arc<hotpatch::Cache>,
    },
}

impl Server {
    async fn new(
        gcxt: &GlobalContext,
        workspace: Workspace<'_>,
        args: &ArgMatches,
    ) -> Result<Self> {
        let sysroot = process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .await
            .map(|out| String::from_utf8(out.stdout))?
            .context("Failed to extract rustc sysroot output")?;

        let target_kind = if args.contains_id("example") {
            TargetKind::ExampleBin
        } else {
            TargetKind::Bin
        };

        let compile_opts = args.compile_options(
            gcxt,
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
                    format!("Failed to find example {example}. \nAvailable examples are:\n{}", examples)
                } else if let Some(bin) = &args.get_one::<String>("bin") {
                    let binaries = target_of_kind(&TargetKind::Bin);
                    format!("Failed to find binary {bin}. \nAvailable binaries are:\n{}", binaries)
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

        let custom_linker = cargo_config.linker(triple.to_string())?;

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

        Ok(Self {
            sysroot: PathBuf::from(sysroot),
            crate_target,
            crate_dir: main_package.manifest_path().parent().unwrap().to_path_buf(),
            workspace_dir: workspace.root_manifest().parent().unwrap().to_path_buf(),
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
            no_default_features: args.get_flag("no-default-features"),
            target_dir,
            custom_linker,
        })
    }

    async fn run(self) -> Result<ExitCode> {
        use notify::Watcher;

        std::fs::create_dir_all(self.exe_dir())?;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(self.link_args_file())?;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(self.link_err_file())?;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(self.rustc_wrapper_args_file())?;

        let (sender, mut receiver) = mpsc::channel(1);

        let mut watcher = notify::recommended_watcher(move |event| {
            let _ = sender.blocking_send(event);
        })?;

        let build = self.build(BuildMode::Fat).await?;
        let mut executable = tokio::process::Command::new(self.main_exe())
            .args(&self.exe_args)
            .spawn()?;

        watcher.watch(
            self.crate_target
                .src_path()
                .path()
                .and_then(|path| path.parent())
                .expect("Get source path"),
            notify::RecursiveMode::Recursive,
        )?;

        let mut server = server::Server::bind().await?;
        let mut connection = server.accept().await?;
        let mut buffer = Vec::new();

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

            let changed_files: BTreeSet<PathBuf> = buffer
                .drain(..n)
                .filter_map(|event| {
                    let event = event.ok()?;

                    let notify::EventKind::Modify(notify::event::ModifyKind::Data(_)) = event.kind
                    else {
                        return None;
                    };

                    let Some(path) = event.paths.first() else {
                        return None;
                    };

                    if path != &path.with_extension("rs") {
                        return None;
                    }

                    Some(event.paths)
                })
                .flatten()
                .collect();

            if changed_files.is_empty() {
                continue;
            }

            if let Err(error) = self.patch(&build, changed_files, &mut connection).await {
                log::error!("{error}");
            }
        }
    }

    async fn build(&self, mode: BuildMode) -> Result<Build> {
        // Run the cargo build to produce our artifacts
        let mut build = self.cargo_build(&mode).await?;

        // Write the build artifacts to the bundle on the disk
        match &mode {
            BuildMode::Thin {
                aslr_reference,
                cache,
                rustc_args,
                ..
            } => {
                self.write_patch(*aslr_reference, &mut build, cache, rustc_args)
                    .await?;
            }

            BuildMode::Fat => {
                if !build.is_fresh {
                    self.write_executable(&build.exe)
                        .await
                        .context("Failed to write main executable")?;

                    log::debug!("Binary created at {}", self.build_dir().display());
                }
            }
        }

        // Populate the patch cache if we're in fat mode
        if matches!(mode, BuildMode::Fat) {
            build.patch_cache = Some(Arc::new(self.create_patch_cache().await?));
        }

        Ok(build)
    }

    async fn patch(
        &self,
        build: &Build,
        changed_files: BTreeSet<PathBuf>,
        connection: &mut server::Connection,
    ) -> Result<()> {
        let patch = self
            .build(BuildMode::Thin {
                rustc_args: build.direct_rustc.clone(),
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

        connection.patch(&jump_table).await?;

        Ok(())
    }

    async fn create_patch_cache(&self) -> Result<hotpatch::Cache> {
        // TODO: Wasm
        let exe = self.main_exe().to_path_buf();

        Ok(hotpatch::Cache::new(&exe, &self.triple)?)
    }

    async fn write_executable(&self, exe: &Path) -> Result<()> {
        // TODO: Wasm
        std::fs::copy(exe, self.main_exe())?;

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

            let Some(Ok(message)) = Message::parse_stream(std::io::Cursor::new(line)).next() else {
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
                            println!("{}", rendered);
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
                Message::BuildFinished(finished) => {
                    if !finished.success {
                        return Err(anyhow::anyhow!(
                            "Cargo build failed, signaled by the compiler. Toggle tracing mode (press `t`) for more information."
                        )
                        .into());
                    }
                }
                _ => {}
            }
        }

        // Accumulate the rustc args from the wrapper, if they exist and can be parsed.
        let mut direct_rustc = rustc::Args::default();
        if let Ok(res) = std::fs::read_to_string(self.rustc_wrapper_args_file()) {
            if let Ok(res) = serde_json::from_str(&res) {
                direct_rustc = res;
            }
        }

        // If there's any warnings from the linker, we should print them out
        if let Ok(linker_warnings) = std::fs::read_to_string(self.link_err_file()) {
            if !linker_warnings.is_empty() {
                if output_location.is_none() {
                    log::error!("Linker warnings: {}", linker_warnings);
                } else {
                    log::debug!("Linker warnings: {}", linker_warnings);
                }
            }
        }

        // Collect the linker args from the and update the rustc args
        direct_rustc.link_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let exe = output_location.context("Cargo build failed - no output location. Toggle tracing mode (press `t`) for more information.")?;

        // Fat builds need to be linked with the fat linker. Would also like to link here for thin builds
        if matches!(mode, BuildMode::Fat) && has_compiled {
            let link_start = SystemTime::now();
            self.run_fat_link(&exe, &direct_rustc).await?;

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
            is_fresh: !has_compiled,
            direct_rustc,
            time_start,
            patch_cache: None,
        })
    }

    async fn write_patch(
        &self,
        aslr_reference: u64,
        build: &mut Build,
        cache: &Arc<hotpatch::Cache>,
        rustc_args: &rustc::Args,
    ) -> anyhow::Result<()> {
        log::debug!(
            "Original builds for patch: {}",
            self.link_args_file().display()
        );

        let raw_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?;

        let args = raw_args.lines().collect::<Vec<_>>();

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
        if self.triple.architecture != Architecture::Wasm32
            && self.triple.architecture != Architecture::Wasm64
        {
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
            for arg in &rustc_args.link_args {
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

        log::trace!("Linking with {:?} using args: {:#?}", linker, object_files);

        // Run the linker directly!
        //
        // We dump its output directly into the patch exe location which is different than how rustc
        // does it since it uses llvm-objcopy into the `target/debug/` folder.
        let res = tokio::process::Command::new(linker)
            .args(object_files.iter())
            .args(dylibs.iter())
            .args(self.thin_link_args(&args)?)
            .args(out_arg)
            .env_clear()
            .envs(rustc_args.envs.iter().map(|(k, v)| (k, v)))
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
        for file in object_files {
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
                // -lxyz
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-framework" || *arg == "-arch" || *arg == "-L" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l") || arg.starts_with("-m") {
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
                // -arch
                // -lxyz
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
                return Err(anyhow::anyhow!("Unsupported platform for thin linking").into());
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

        if let Some(vale) = extract_value("-isysroot") {
            out_args.push("-isysroot".to_string());
            out_args.push(vale);
        }

        Ok(out_args)
    }

    fn main_exe(&self) -> PathBuf {
        self.exe_dir().join(self.platform_exe_name())
    }

    fn executable_name(&self) -> &str {
        &self.crate_target.name()
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
    pub(crate) async fn run_fat_link(&self, exe: &Path, rustc_args: &rustc::Args) -> Result<()> {
        use uuid::Uuid;

        // Filter out the rlib files from the arguments
        let rlibs = rustc_args
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
                    log::trace!("Skipping rlib: {:?}", rlib);
                    continue;
                }

                log::trace!("Adding rlib to staticlib: {:?}", rlib);

                let rlib_contents = std::fs::read(rlib)?;
                let mut reader = ar::Archive::new(std::io::Cursor::new(rlib_contents));
                while let Some(Ok(object_file)) = reader.next_entry() {
                    let name = std::str::from_utf8(object_file.header().identifier()).unwrap();
                    if name.ends_with(".rmeta") {
                        continue;
                    }

                    if object_file.header().size() == 0 {
                        continue;
                    }

                    // rlibs might contain dlls/sos/lib files which we don't want to include
                    if name.ends_with(".dll")
                        || name.ends_with(".so")
                        || name.ends_with(".lib")
                        || name.ends_with(".dylib")
                    {
                        compiler_rlibs.push(rlib.to_owned());
                        continue;
                    }

                    if !(name.ends_with(".o") || name.ends_with(".obj")) {
                        log::debug!("Unknown object file in rlib: {:?}", name);
                    }

                    archive_has_contents = true;
                    out_ar
                        .append(&object_file.header().clone(), object_file)
                        .context("Failed to add object file to archive")?;
                }
            }

            let bytes = out_ar.into_inner().context("Failed to finalize archive")?;
            std::fs::write(&out_ar_path, bytes).context("Failed to write archive")?;
            log::debug!("Wrote fat archive to {:?}", out_ar_path);

            // Run the ranlib command to index the archive. This slows down this process a bit,
            // but is necessary for some linkers to work properly.
            // We ignore its error in case it doesn't recognize the architecture
            if self.linker_flavor() == LinkerFlavor::Darwin {
                if let Some(ranlib) = select_ranlib() {
                    _ = tokio::process::Command::new(ranlib)
                        .arg(&out_ar_path)
                        .output()
                        .await;
                }
            }
        }

        compiler_rlibs.dedup();

        // We're going to replace the first rlib in the args with our fat archive
        // And then remove the rest of the rlibs
        //
        // We also need to insert the -force_load flag to force the linker to load the archive
        let mut args = rustc_args.link_args.clone();

        if let Some(last_object) = args.iter().rposition(|arg| arg.ends_with(".o")) {
            if archive_has_contents {
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
        }

        // Add custom args to the linkers
        match self.linker_flavor() {
            LinkerFlavor::Gnu => {
                // Export `main` so subsecond can use it for a reference point
                args.push("-Wl,--export-dynamic-symbol,main".to_string());
            }
            LinkerFlavor::Darwin => {
                // `-all_load` is an extra step to ensure that all symbols are loaded (different than force_load)
                args.push("-Wl,-all_load".to_string());
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
            args.remove(idx + 1);
            args.remove(idx);
        }

        // same but windows support
        if let Some(idx) = args.iter().position(|arg| arg.starts_with("/OUT")) {
            args.remove(idx);
        }

        // TODO
        // We want to go through wasm-ld directly, so we need to remove the -flavor flag
        // if self.platform == Platform::Web {
        //     let flavor_idx = args.iter().position(|arg| *arg == "-flavor").unwrap();
        //     args.remove(flavor_idx + 1);
        //     args.remove(flavor_idx);
        // }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;

        log::trace!("Fat linking with args: {:?} {:#?}", linker, args);
        log::trace!("Fat linking with env: {:#?}", rustc_args.envs);

        // Run the linker directly!
        let out_arg = match self.triple.operating_system {
            OperatingSystem::Windows => vec![format!("/OUT:{}", exe.display())],
            _ => vec!["-o".to_string(), exe.display().to_string()],
        };

        let res = tokio::process::Command::new(linker)
            .args(args.iter().skip(1))
            .args(out_arg)
            .env_clear()
            .envs(rustc_args.envs.iter().map(|(k, v)| (k, v)))
            .output()
            .await?;

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            if !res.status.success() {
                log::error!("Failed to generate fat binary: {}", errs.trim());
            } else {
                log::trace!("Warnings during fat linking: {}", errs.trim());
            }
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
        ) {
            if let Ok(linker) = std::env::var("DX_HOST_LINKER") {
                return Ok(PathBuf::from(linker));
            }
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
            BuildMode::Thin { rustc_args, .. } => {
                let mut cmd = tokio::process::Command::new("rustc");
                cmd.current_dir(&self.workspace_dir);
                cmd.env_clear();
                cmd.args(rustc_args.args[1..].iter());
                cmd.env_remove("RUSTC_WORKSPACE_WRAPPER");
                cmd.env_remove("RUSTC_WRAPPER");
                cmd.env_remove(rustc::DX_RUSTC_WRAPPER_ENV_VAR);
                cmd.envs(self.cargo_build_env_vars(mode)?);
                cmd.arg(format!("-Clinker={}", path_to_me()?.display()));

                // TODO
                // if self.platform == Platform::Web {
                //     cmd.arg("-Crelocation-model=pic");
                // }

                log::debug!("Direct rustc: {:#?}", cmd);

                cmd.envs(rustc_args.envs.iter().cloned());

                // tracing::trace!("Setting env vars: {:#?}", rustc_args.envs);

                Ok(cmd)
            }

            // For Base and Fat builds, we use a regular cargo setup, but we might need to intercept
            // rustc itself in case we're hot-patching and need a reliable rustc environment to
            // continuously recompile the top-level crate with.
            //
            // In the future, when we support hot-patching *all* workspace crates, we will need to
            // make use of the RUSTC_WORKSPACE_WRAPPER environment variable instead of RUSTC_WRAPPER
            // and then keep track of env and args on a per-crate basis.
            //
            // We've also had a number of issues with incorrect canonicalization when passing paths
            // through envs on windows, hence the frequent use of dunce::canonicalize.
            _ => {
                let mut cmd = tokio::process::Command::new("cargo");

                cmd.arg("rustc")
                    .current_dir(&self.crate_dir)
                    .arg("--message-format")
                    .arg("json-diagnostic-rendered-ansi")
                    .arg("--color")
                    .arg("always")
                    .args(self.cargo_build_arguments(mode))
                    .envs(self.cargo_build_env_vars(mode)?);

                if mode == &BuildMode::Fat {
                    cmd.env(
                        rustc::DX_RUSTC_WRAPPER_ENV_VAR,
                        dunce::canonicalize(self.rustc_wrapper_args_file())
                            .unwrap()
                            .display()
                            .to_string(),
                    );
                    cmd.env("RUSTC_WRAPPER", path_to_me()?.display().to_string());
                }

                log::debug!("Cargo: {:#?}", cmd);

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
        cargo_args.push("--verbose".to_string());

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
            // if self.platform == Platform::Web
            //     || self.triple.operating_system == OperatingSystem::Wasi
            // {
            //     cargo_args.push("-Ctarget-cpu=mvp".into());
            //     cargo_args.push("-Clink-arg=--no-gc-sections".into());
            //     cargo_args.push("-Clink-arg=--growable-table".into());
            //     cargo_args.push("-Clink-arg=--export-table".into());
            //     cargo_args.push("-Clink-arg=--export-memory".into());
            //     cargo_args.push("-Clink-arg=--emit-relocs".into());
            //     cargo_args.push("-Clink-arg=--export=__stack_pointer".into());
            //     cargo_args.push("-Clink-arg=--export=__heap_base".into());
            //     cargo_args.push("-Clink-arg=--export=__data_end".into());
            // }
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

    fn rustc_wrapper_args_file(&self) -> PathBuf {
        self.exe_dir().join("rustc_wrapper_args.txt")
    }
}

fn select_ranlib() -> Option<PathBuf> {
    // prefer the modern llvm-ranlib if they have it
    which::which("llvm-ranlib")
        .or_else(|_| which::which("ranlib"))
        .ok()
}

fn path_to_me() -> Result<PathBuf> {
    Ok(
        dunce::canonicalize(std::env::current_exe().context("Failed to find cargo-hot")?)
            .context("Failed to find cargo-hot")?,
    )
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

    return n;
}
