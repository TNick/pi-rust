//! Isolated native-plugin execution over a versioned worker process protocol.
//!
//! The host never maps a plugin cdylib into its own address space. It starts a
//! caller-selected worker executable, owns the child process tree, and exchanges
//! newline-delimited [`protocol::WorkerFrame`] values over piped stdio. A worker
//! loads plugins and executes registered tools inside its own process. Plugin
//! `runtime_action` calls are relayed as [`protocol::WorkerFrame::HostCall`]
//! frames and resolved only by the host's [`WorkerHostHandler`]; the generic
//! [`AllowlistedHostHandler`] can enforce a configured capability allowlist.
//!
//! OS sandboxing is an explicit launch policy. A sandboxed request is rejected
//! unless the selected launcher reports filesystem, network, and secret
//! guarantees. The default [`HostProcessLauncher`] fails closed for sandboxed
//! requests; see `ISOLATED_WORKER_SANDBOX.md` for platform coverage.
//!
//! The launcher is injected by the host so the product can apply its own OS
//! sandbox/credential policy without adding it to this crate.

use std::path::PathBuf;

pub mod client;
pub mod plugin;
pub mod protocol;
pub mod worker;

pub use client::{
    AllowlistedHostHandler, HostProcessLauncher, IsolatedWorker, LinuxBubblewrapLauncher,
    WorkerError, WorkerFilesystemPolicy, WorkerHostHandler, WorkerIsolation, WorkerLaunchSpec,
    WorkerLauncher, WorkerNetworkPolicy, WorkerSandboxGuarantees, WorkerSandboxPolicy,
    WorkerSecretPolicy,
};
pub use plugin::{run_plugin_worker, PluginWorker};
pub use protocol::{WorkerFrame, WorkerToolSpec, WORKER_PROTOCOL};
pub use worker::{
    serve_worker, serve_worker_stdio, WorkerFrameWriter, WorkerHandler, WorkerHostClient,
};

/// Loads plugins selected by a host into a worker process and serves the host.
///
/// # Parameters
/// - `directories`: Authorized plugin directories.
/// - `files`: Authorized explicit plugin files.
///
/// # Returns
/// `Ok(())` after orderly host shutdown.
///
/// # Errors
/// Returns a load or protocol error.
pub fn run_worker(directories: &[PathBuf], files: &[PathBuf]) -> Result<(), String> {
    plugin::run_plugin_worker(directories, files)
}
