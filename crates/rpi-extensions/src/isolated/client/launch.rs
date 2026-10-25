//! Defines host-selected worker launching, sandbox policy, and host capability policy.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

/// Required isolation level for one plugin worker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkerIsolation {
    /// The host accepts a direct child process with environment clearing and
    /// process-tree supervision only. This is not an OS sandbox.
    #[default]
    Unisolated,
    /// The host requires OS-enforced filesystem, network, and secret limits.
    Sandboxed,
}

/// Filesystem restrictions requested from an OS sandbox launcher.
#[derive(Clone, Debug, Default)]
pub struct WorkerFilesystemPolicy {
    /// Absolute paths the worker may read.
    pub read_roots: Vec<PathBuf>,
    /// Absolute paths the worker may write.
    pub write_roots: Vec<PathBuf>,
}

/// Network restriction requested from an OS sandbox launcher.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkerNetworkPolicy {
    /// Deny all network access.
    #[default]
    Deny,
    /// Permit network access.
    Allow,
}

/// Secret/environment restriction requested from an OS sandbox launcher.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkerSecretPolicy {
    /// Clear the host environment and pass only the declared worker environment.
    #[default]
    DeclaredOnly,
    /// Clear the host environment and pass no environment values.
    DenyAll,
}

/// Complete sandbox policy requested by the host.
#[derive(Clone, Debug, Default)]
pub struct WorkerSandboxPolicy {
    /// Required isolation level.
    pub isolation: WorkerIsolation,
    /// Filesystem roots permitted to the worker.
    pub filesystem: WorkerFilesystemPolicy,
    /// Network access policy.
    pub network: WorkerNetworkPolicy,
    /// Environment/secret policy.
    pub secrets: WorkerSecretPolicy,
}

impl WorkerSandboxPolicy {
    /// Builds a sandboxed policy with explicit filesystem roots.
    ///
    /// # Parameters
    /// - `read_roots`: Absolute paths the worker may read.
    /// - `write_roots`: Absolute paths the worker may write.
    ///
    /// # Returns
    /// A policy that denies network access and permits only declared
    /// environment values.
    pub fn sandboxed(read_roots: Vec<PathBuf>, write_roots: Vec<PathBuf>) -> Self {
        Self {
            isolation: WorkerIsolation::Sandboxed,
            filesystem: WorkerFilesystemPolicy {
                read_roots,
                write_roots,
            },
            network: WorkerNetworkPolicy::Deny,
            secrets: WorkerSecretPolicy::DeclaredOnly,
        }
    }

    /// Reports whether OS sandboxing is required.
    ///
    /// # Returns
    /// `true` for [`WorkerIsolation::Sandboxed`].
    pub fn is_sandboxed(&self) -> bool {
        self.isolation == WorkerIsolation::Sandboxed
    }
}

/// What a launcher actually enforces for one worker process.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkerSandboxGuarantees {
    /// Whether the launcher enforces the requested filesystem roots.
    pub filesystem: bool,
    /// Whether the launcher enforces the requested network policy.
    pub network: bool,
    /// Whether the launcher enforces the requested environment/secret policy.
    pub secrets: bool,
    /// Platform implementation providing the guarantee.
    pub platform: &'static str,
}

impl WorkerSandboxGuarantees {
    /// Returns a no-sandbox guarantee set.
    ///
    /// # Parameters
    /// - `platform`: Platform implementation name.
    ///
    /// # Returns
    /// A guarantee set that satisfies no sandbox requirement.
    pub fn none(platform: &'static str) -> Self {
        Self {
            filesystem: false,
            network: false,
            secrets: false,
            platform,
        }
    }

    /// Reports whether these guarantees satisfy one sandbox policy.
    ///
    /// # Parameters
    /// - `policy`: Host-requested sandbox policy.
    ///
    /// # Returns
    /// `true` for an unisolated policy, or when all three OS restrictions are
    /// claimed for a sandboxed policy.
    pub fn enforces(&self, policy: &WorkerSandboxPolicy) -> bool {
        !policy.is_sandboxed() || (self.filesystem && self.network && self.secrets)
    }
}

/// Describes how the host should start one plugin worker process.
#[derive(Clone, Debug)]
pub struct WorkerLaunchSpec {
    /// Worker executable selected by the host.
    pub executable: PathBuf,
    /// Fixed worker arguments.
    pub args: Vec<String>,
    /// Additional environment values.
    pub environment: BTreeMap<String, String>,
    /// Optional working directory.
    pub working_directory: Option<PathBuf>,
    /// Whether the worker inherits the host process environment.
    pub inherit_environment: bool,
    /// Requested OS sandbox policy.
    pub sandbox: WorkerSandboxPolicy,
    /// Maximum time allowed for worker readiness.
    pub startup_timeout: Duration,
    /// Maximum time allowed for one tool invocation.
    pub request_timeout: Duration,
}

impl Default for WorkerLaunchSpec {
    /// Creates a launch spec with bounded default timeouts and no sandbox claim.
    fn default() -> Self {
        Self {
            executable: PathBuf::new(),
            args: Vec::new(),
            environment: BTreeMap::new(),
            working_directory: None,
            inherit_environment: false,
            sandbox: WorkerSandboxPolicy::default(),
            startup_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(120),
        }
    }
}

/// Host-injected worker command launcher.
pub trait WorkerLauncher: Send + Sync {
    /// Returns the OS restrictions this launcher actually enforces.
    ///
    /// The default is no sandbox guarantee. Launchers that support
    /// [`WorkerIsolation::Sandboxed`] must report every enforced restriction.
    ///
    /// # Returns
    /// The launcher's platform-specific guarantee set.
    fn guarantees(&self) -> WorkerSandboxGuarantees {
        WorkerSandboxGuarantees::none(std::env::consts::OS)
    }

    /// Builds the command used to start one worker.
    ///
    /// # Parameters
    /// - `spec`: Host-selected executable, environment, and sandbox policy.
    ///
    /// # Returns
    /// A configured command. The client adds piped stdio and process-tree
    /// supervision before spawning.
    ///
    /// # Errors
    /// Returns an error when the launcher cannot construct the command or
    /// cannot enforce the requested sandbox policy.
    fn command(&self, spec: &WorkerLaunchSpec) -> std::io::Result<Command>;
}

/// Default direct launcher. It fails closed for every sandboxed request.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostProcessLauncher;

impl WorkerLauncher for HostProcessLauncher {
    /// Builds a direct host command for an unisolated worker.
    ///
    /// # Errors
    /// Returns an unsupported error when the spec requests an OS sandbox.
    fn command(&self, spec: &WorkerLaunchSpec) -> std::io::Result<Command> {
        if spec.sandbox.is_sandboxed() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "no OS sandbox is available for this worker launcher",
            ));
        }
        let mut command = Command::new(&spec.executable);
        if !spec.inherit_environment || spec.sandbox.secrets == WorkerSecretPolicy::DenyAll {
            command.env_clear();
        }
        if spec.sandbox.secrets == WorkerSecretPolicy::DeclaredOnly {
            command.envs(&spec.environment);
        }
        command.args(&spec.args);
        if let Some(directory) = &spec.working_directory {
            command.current_dir(directory);
        }
        Ok(command)
    }
}

/// Linux bubblewrap launcher with namespace-enforced OS isolation.
///
/// It requires `bwrap` on the host and a kernel that permits the requested
/// namespaces. Missing `bwrap`, unsupported namespaces, or invalid roots fail
/// the launch; the launcher never falls back to an unsandboxed process. `/proc`
/// is left unmounted because some hosts prohibit mounting proc inside a user
/// namespace; the isolated worker protocol does not require it.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinuxBubblewrapLauncher;

impl WorkerLauncher for LinuxBubblewrapLauncher {
    /// Reports the filesystem, network, and environment restrictions enforced
    /// by bubblewrap.
    fn guarantees(&self) -> WorkerSandboxGuarantees {
        WorkerSandboxGuarantees {
            filesystem: true,
            network: true,
            secrets: true,
            platform: "linux-bubblewrap",
        }
    }

    /// Builds a bubblewrap sandbox command.
    ///
    /// # Errors
    /// Returns an unsupported error on non-Linux hosts, for inherited host
    /// environments, or for relative filesystem roots. Runtime failures from a
    /// missing/incompatible `bwrap` surface as spawn errors.
    fn command(&self, spec: &WorkerLaunchSpec) -> io::Result<Command> {
        if !cfg!(target_os = "linux") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "bubblewrap launcher is supported only on Linux",
            ));
        }
        if !spec.sandbox.is_sandboxed() {
            return HostProcessLauncher.command(spec);
        }
        if spec.inherit_environment {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "sandboxed worker cannot inherit the host environment",
            ));
        }
        validate_roots(&spec.sandbox.filesystem)?;
        let mut command = Command::new("bwrap");
        command
            .arg("--die-with-parent")
            .arg("--new-session")
            .arg("--unshare-all")
            .arg("--cap-drop")
            .arg("ALL")
            .arg("--clearenv")
            .arg("--dev")
            .arg("/dev")
            .arg("--tmpfs")
            .arg("/tmp");
        let mut write_roots = BTreeSet::new();
        for root in &spec.sandbox.filesystem.write_roots {
            if write_roots.insert(root.clone()) {
                command.arg("--bind").arg(root).arg(root);
            }
        }
        for root in &spec.sandbox.filesystem.read_roots {
            if !write_roots.contains(root) {
                command.arg("--ro-bind").arg(root).arg(root);
            }
        }
        if spec.sandbox.network == WorkerNetworkPolicy::Allow {
            command.arg("--share-net");
        }
        if spec.sandbox.secrets == WorkerSecretPolicy::DeclaredOnly {
            for (name, value) in &spec.environment {
                command.arg("--setenv").arg(name).arg(value);
            }
        }
        let directory = spec
            .working_directory
            .as_deref()
            .unwrap_or_else(|| Path::new("/tmp"));
        command.arg("--chdir").arg(directory);
        command.arg("--").arg(&spec.executable).args(&spec.args);
        Ok(command)
    }
}

/// Rejects relative or empty filesystem roots before invoking an OS sandbox.
///
/// # Parameters
/// - `filesystem`: Requested filesystem roots.
///
/// # Returns
/// `Ok(())` when every configured root is absolute and non-empty.
///
/// # Errors
/// Returns an invalid-input error for a root that cannot be safely bound.
fn validate_roots(filesystem: &WorkerFilesystemPolicy) -> io::Result<()> {
    if filesystem
        .read_roots
        .iter()
        .chain(&filesystem.write_roots)
        .any(|root| root.as_os_str().is_empty() || !root.is_absolute())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sandbox filesystem roots must be absolute and non-empty",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "launch_tests.rs"]
mod tests;

/// Host policy for worker-initiated capability calls.
pub trait WorkerHostHandler: Send + Sync {
    /// Handles one worker capability request.
    ///
    /// # Parameters
    /// - `capability`: Stable capability identifier.
    /// - `arguments`: Capability arguments.
    ///
    /// # Returns
    /// A bounded result for the worker.
    ///
    /// # Errors
    /// Returns a safe denial or failure text.
    fn call(&self, capability: &str, arguments: &Value) -> Result<Value, String>;
}

/// Host-side capability allowlist for worker runtime actions.
pub struct AllowlistedHostHandler {
    /// Capability names the host permits this worker to request.
    allowed: BTreeSet<String>,
    /// Host policy that performs an allowed capability.
    inner: Arc<dyn WorkerHostHandler>,
}

impl AllowlistedHostHandler {
    /// Creates a fail-closed capability allowlist wrapper.
    ///
    /// # Parameters
    /// - `allowed`: Exact capability names the worker may request.
    /// - `inner`: Host handler that executes an allowed capability.
    ///
    /// # Returns
    /// A host handler that denies every unlisted capability.
    pub fn new(allowed: BTreeSet<String>, inner: Arc<dyn WorkerHostHandler>) -> Self {
        Self { allowed, inner }
    }
}

impl WorkerHostHandler for AllowlistedHostHandler {
    /// Enforces the configured allowlist before invoking the host handler.
    ///
    /// # Parameters
    /// - `capability`: Worker-requested capability.
    /// - `arguments`: Capability arguments.
    ///
    /// # Returns
    /// The inner host result for an allowed capability.
    ///
    /// # Errors
    /// Returns a safe denial for an unlisted capability.
    fn call(&self, capability: &str, arguments: &Value) -> Result<Value, String> {
        if !self.allowed.contains(capability) {
            return Err("plugin capability is not authorized for this worker".to_owned());
        }
        self.inner.call(capability, arguments)
    }
}
