//! Starts, owns, and talks to one isolated plugin worker process.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rpi_plugin_sdk::process::ProcessTree;
use serde_json::Value;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use super::protocol::{WorkerFrame, WorkerToolSpec};

mod launch;
mod reader;
pub use launch::{
    AllowlistedHostHandler, HostProcessLauncher, LinuxBubblewrapLauncher, WorkerFilesystemPolicy,
    WorkerHostHandler, WorkerIsolation, WorkerLaunchSpec, WorkerLauncher, WorkerNetworkPolicy,
    WorkerSandboxGuarantees, WorkerSandboxPolicy, WorkerSecretPolicy,
};
use reader::read_worker;

/// Failure from one isolated worker process.
#[derive(Debug, Error)]
pub enum WorkerError {
    /// The worker process could not be started.
    #[error("start plugin worker: {0}")]
    Spawn(String),
    /// The worker did not become ready in time.
    #[error("plugin worker startup timed out")]
    StartupTimeout,
    /// A worker request exceeded its deadline.
    #[error("plugin worker request timed out")]
    RequestTimeout,
    /// The worker connection closed.
    #[error("plugin worker closed: {0}")]
    Closed(String),
    /// The worker reported a tool failure.
    #[error("plugin tool failed: {0}")]
    Tool(String),
    /// The requested tool was not advertised by the worker.
    #[error("plugin tool is not authorized: {0}")]
    Unauthorized(String),
    /// The caller cancelled the request.
    #[error("plugin tool cancelled")]
    Cancelled,
    /// The requested OS sandbox cannot be enforced.
    #[error("plugin worker sandbox unavailable: {0}")]
    Sandbox(String),
    /// Worker framing failed.
    #[error("plugin worker protocol: {0}")]
    Protocol(String),
}

type Pending = Arc<Mutex<HashMap<u64, mpsc::Sender<Result<Value, String>>>>>;

/// Owns one isolated plugin worker process and its serialized IPC channel.
pub struct IsolatedWorker {
    child: Child,
    process_tree: Option<ProcessTree>,
    input: Arc<Mutex<ChildStdin>>,
    pending: Pending,
    closed: Arc<Mutex<Option<String>>>,
    next_id: AtomicU64,
    tools: Vec<WorkerToolSpec>,
    request_timeout: Duration,
}

impl IsolatedWorker {
    /// Starts a worker and waits for its versioned readiness frame.
    ///
    /// # Parameters
    /// - `spec`: Executable, environment, and timeouts.
    /// - `launcher`: Host-injected process launcher.
    /// - `host_handler`: Optional host capability policy.
    ///
    /// # Returns
    /// A ready worker with its advertised tools.
    ///
    /// # Errors
    /// Returns a spawn, readiness, or protocol error. A failed start kills the
    /// already-spawned process tree.
    pub fn start(
        spec: &WorkerLaunchSpec,
        launcher: &dyn WorkerLauncher,
        host_handler: Option<Arc<dyn WorkerHostHandler>>,
    ) -> Result<Self, WorkerError> {
        let guarantees = launcher.guarantees();
        if !guarantees.enforces(&spec.sandbox) {
            return Err(WorkerError::Sandbox(format!(
                "launcher `{}` does not enforce the requested filesystem, network, and environment restrictions",
                guarantees.platform
            )));
        }
        let mut command = launcher
            .command(spec)
            .map_err(|error| WorkerError::Spawn(error.to_string()))?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        ProcessTree::prepare(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| WorkerError::Spawn(error.to_string()))?;
        let process_tree =
            ProcessTree::attach(&child).map_err(|error| WorkerError::Spawn(error))?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::Spawn("worker stdin unavailable".to_owned()))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError::Spawn("worker stdout unavailable".to_owned()))?;
        let errors = child.stderr.take();

        let input = Arc::new(Mutex::new(input));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let closed = Arc::new(Mutex::new(None));
        let (hello_sender, hello_receiver) = mpsc::channel();

        if let Some(errors) = errors {
            thread::spawn(move || {
                for line in BufReader::new(errors).lines().map_while(Result::ok) {
                    tracing::warn!(target: "rpi_extensions::worker", "{line}");
                }
            });
        }
        {
            let input = Arc::clone(&input);
            let pending = Arc::clone(&pending);
            let closed = Arc::clone(&closed);
            thread::spawn(move || {
                read_worker(output, input, pending, closed, hello_sender, host_handler);
            });
        }

        let tools = match hello_receiver.recv_timeout(spec.startup_timeout) {
            Ok(Ok(tools)) => tools,
            Ok(Err(error)) => {
                process_tree.kill();
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            Err(_) => {
                process_tree.kill();
                let _ = child.kill();
                let _ = child.wait();
                return Err(WorkerError::StartupTimeout);
            }
        };
        Ok(Self {
            child,
            process_tree: Some(process_tree),
            input,
            pending,
            closed,
            next_id: AtomicU64::new(1),
            tools,
            request_timeout: spec.request_timeout,
        })
    }

    /// Returns the tools advertised by this worker.
    pub fn tools(&self) -> &[WorkerToolSpec] {
        &self.tools
    }

    /// Executes one advertised tool in the worker process.
    ///
    /// # Parameters
    /// - `tool`: Advertised tool identifier.
    /// - `arguments`: Tool arguments.
    ///
    /// # Returns
    /// The worker-produced JSON result.
    ///
    /// # Errors
    /// Returns an authorization, protocol, timeout, cancellation, or tool error.
    pub fn call_tool(&self, tool: &str, arguments: Value) -> Result<Value, WorkerError> {
        self.call_tool_cancellable(tool, arguments, None)
    }

    /// Executes one advertised tool with optional cancellation.
    ///
    /// # Parameters
    /// - `tool`: Advertised tool identifier.
    /// - `arguments`: Tool arguments.
    /// - `cancellation`: Optional host cancellation token.
    ///
    /// # Returns
    /// The worker-produced JSON result.
    ///
    /// # Errors
    /// Returns an authorization, protocol, timeout, cancellation, or tool error.
    pub fn call_tool_cancellable(
        &self,
        tool: &str,
        arguments: Value,
        cancellation: Option<&CancellationToken>,
    ) -> Result<Value, WorkerError> {
        if !self.tools.iter().any(|advertised| advertised.name == tool) {
            return Err(WorkerError::Unauthorized(tool.to_owned()));
        }
        if let Some(error) = self.closed.lock().ok().and_then(|closed| closed.clone()) {
            return Err(WorkerError::Closed(error));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::channel();
        if let Ok(mut pending) = self.pending.lock() {
            pending.insert(id, sender);
        }
        if let Err(error) = self.send(&WorkerFrame::ToolCall {
            id,
            tool: tool.to_owned(),
            arguments,
        }) {
            self.forget(id);
            return Err(error);
        }
        let deadline = Instant::now()
            .checked_add(self.request_timeout)
            .unwrap_or_else(Instant::now);
        let mut cancelled = false;
        loop {
            if cancellation.is_some_and(|token| token.is_cancelled()) && !cancelled {
                cancelled = true;
                let _ = self.send(&WorkerFrame::Cancel { id });
            }
            if cancelled {
                self.forget(id);
                return Err(WorkerError::Cancelled);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.forget(id);
                self.terminate("plugin worker request timed out; process terminated");
                return Err(WorkerError::RequestTimeout);
            }
            let wait = remaining.min(Duration::from_millis(25));
            match receiver.recv_timeout(wait) {
                Ok(Ok(result)) => return Ok(result),
                Ok(Err(error)) => return Err(WorkerError::Tool(error)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let error = self
                        .closed
                        .lock()
                        .ok()
                        .and_then(|closed| closed.clone())
                        .unwrap_or_else(|| "worker connection closed".to_owned());
                    return Err(WorkerError::Closed(error));
                }
            }
        }
    }

    /// Stops the worker process and its descendants.
    ///
    /// # Returns
    /// `Ok(())` after the process is gone or the shutdown request was sent.
    ///
    /// # Errors
    /// Returns an error when the shutdown frame cannot be written.
    pub fn shutdown(&mut self) -> Result<(), WorkerError> {
        let sent = self.send(&WorkerFrame::Shutdown);
        for _ in 0..20 {
            if self.child.try_wait().ok().flatten().is_some() {
                self.kill_tree();
                return sent;
            }
            thread::sleep(Duration::from_millis(25));
        }
        self.kill_tree();
        let _ = self.child.kill();
        let _ = self.child.wait();
        sent
    }

    /// Sends one frame to the worker.
    ///
    /// # Errors
    /// Returns an error when the frame cannot be encoded or written.
    fn send(&self, frame: &WorkerFrame) -> Result<(), WorkerError> {
        let mut bytes = frame
            .encode()
            .map_err(|error| WorkerError::Protocol(error.to_string()))?;
        bytes.push(b'\n');
        let Ok(mut input) = self.input.lock() else {
            return Err(WorkerError::Closed("worker stdin poisoned".to_owned()));
        };
        input
            .write_all(&bytes)
            .and_then(|()| input.flush())
            .map_err(|error| WorkerError::Closed(error.to_string()))
    }

    /// Removes one pending correlation entry.
    ///
    /// # Parameters
    /// - `id`: Correlation identity.
    fn forget(&self, id: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(&id);
        }
    }

    /// Terminates the worker's supervised process tree.
    fn kill_tree(&self) {
        if let Some(tree) = &self.process_tree {
            tree.kill();
        }
    }

    /// Marks the worker unusable, terminates its process tree, and fails every
    /// outstanding request.
    ///
    /// # Parameters
    /// - `message`: Safe termination reason.
    fn terminate(&self, message: &str) {
        self.kill_tree();
        if let Ok(mut closed) = self.closed.lock() {
            *closed = Some(message.to_owned());
        }
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(message.to_owned()));
            }
        }
    }
}

impl Drop for IsolatedWorker {
    /// Shuts down and reaps the worker process.
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
