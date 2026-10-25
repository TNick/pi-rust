//! Worker-side JSON-lines loop and host-capability bridge.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::protocol::{
    WorkerCommandSpec, WorkerFrame, WorkerProviderRequest, WorkerProviderResponse,
    WorkerProviderSpec, WorkerToolSpec,
};

#[cfg(test)]
#[path = "worker/tests.rs"]
mod tests;

/// Writes one worker frame to the host command stream.
pub trait WorkerFrameWriter: Send + Sync {
    /// Writes and flushes one frame.
    ///
    /// # Parameters
    /// - `frame`: Frame to serialize.
    ///
    /// # Returns
    /// `Ok(())` after the frame is flushed.
    ///
    /// # Errors
    /// Returns an I/O error when the stream is unavailable.
    fn write_frame(&self, frame: &WorkerFrame) -> std::io::Result<()>;
}

/// Serializes frames through one shared writer.
struct SharedWriter<W>(Arc<Mutex<W>>);

impl<W: Write + Send + 'static> WorkerFrameWriter for SharedWriter<W> {
    /// Writes and flushes one frame through the shared writer.
    ///
    /// # Errors
    /// Returns an I/O error when the stream is poisoned or unwritable.
    fn write_frame(&self, frame: &WorkerFrame) -> std::io::Result<()> {
        let mut bytes = frame
            .encode()
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        bytes.push(b'\n');
        let Ok(mut output) = self.0.lock() else {
            return Err(std::io::Error::other("worker output poisoned"));
        };
        output.write_all(&bytes)?;
        output.flush()
    }
}

/// Worker-side client for host-mediated capability calls.
pub struct WorkerHostClient {
    /// Response stream shared with the main worker loop.
    writer: Arc<dyn WorkerFrameWriter>,
    /// Outstanding host-call response senders.
    pending: Mutex<HashMap<u64, mpsc::Sender<Result<Value, String>>>>,
    /// Next host-call correlation identity.
    next_id: AtomicU64,
    /// Maximum time to wait for one host reply.
    timeout: Duration,
}

impl WorkerHostClient {
    /// Creates a host-call client over the worker response stream.
    ///
    /// # Parameters
    /// - `writer`: Shared worker frame writer.
    ///
    /// # Returns
    /// A client with a bounded default reply timeout.
    pub fn new(writer: Arc<dyn WorkerFrameWriter>) -> Self {
        Self {
            writer,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            timeout: Duration::from_secs(60),
        }
    }

    /// Sets the host-reply timeout.
    ///
    /// # Parameters
    /// - `timeout`: Maximum reply wait.
    ///
    /// # Returns
    /// The updated client.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Requests one capability from the host.
    ///
    /// # Parameters
    /// - `capability`: Stable capability identifier.
    /// - `arguments`: Capability arguments.
    ///
    /// # Returns
    /// The host result.
    ///
    /// # Errors
    /// Returns a bounded denial, timeout, or write failure.
    pub fn call(&self, capability: &str, arguments: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::channel();
        if let Ok(mut pending) = self.pending.lock() {
            pending.insert(id, sender);
        }
        if let Err(error) = self.writer.write_frame(&WorkerFrame::HostCall {
            id,
            capability: capability.to_owned(),
            arguments,
        }) {
            if let Ok(mut pending) = self.pending.lock() {
                pending.remove(&id);
            }
            return Err(error.to_string());
        }
        match receiver.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(_) => {
                if let Ok(mut pending) = self.pending.lock() {
                    pending.remove(&id);
                }
                Err("host capability request timed out".to_owned())
            }
        }
    }

    /// Resolves one host reply.
    ///
    /// # Parameters
    /// - `id`: Host-call correlation identity.
    /// - `result`: Host result or denial.
    fn resolve(&self, id: u64, result: Result<Value, String>) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(sender) = pending.remove(&id) {
                let _ = sender.send(result);
            }
        }
    }
}

/// Worker-side handler executed outside the host process.
pub trait WorkerHandler: Send + Sync {
    /// Returns the tools this worker advertises to its host.
    fn tools(&self) -> Vec<WorkerToolSpec>;

    /// Returns providers and models this worker is able to execute.
    ///
    /// # Returns
    /// Credential-free provider registrations. Workers without provider
    /// support may use the empty default.
    fn providers(&self) -> Vec<WorkerProviderSpec> {
        Vec::new()
    }

    /// Returns slash commands the host may expose and invoke.
    fn commands(&self) -> Vec<WorkerCommandSpec> {
        Vec::new()
    }

    /// Prepares worker state with the host-capability client.
    ///
    /// # Parameters
    /// - `host`: Host-capability bridge for plugin runtime actions.
    ///
    /// # Errors
    /// Returns a safe setup failure.
    fn start(&self, _host: Arc<WorkerHostClient>) -> Result<(), String> {
        Ok(())
    }

    /// Executes one advertised tool.
    ///
    /// Cancellation is cooperative: the worker driver cancels the token and the
    /// plugin driver may cancel its native call. Native FFI that never yields
    /// cannot be preempted; the host must terminate the process at its deadline.
    ///
    /// # Parameters
    /// - `tool`: Advertised tool identifier.
    /// - `arguments`: Tool arguments.
    /// - `cancellation`: Shared host cancellation token.
    ///
    /// # Returns
    /// A bounded JSON result.
    ///
    /// # Errors
    /// Returns a safe failure text.
    fn call_tool(
        &self,
        tool: &str,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<Value, String>;

    /// Executes one provider request inside the worker process.
    ///
    /// # Parameters
    /// - `request`: Host-authorized provider, model, context, options, and
    ///   request-only credential.
    /// - `cancellation`: Shared host cancellation token.
    ///
    /// # Returns
    /// A completed assistant response.
    ///
    /// # Errors
    /// Returns a safe provider failure.
    fn call_provider(
        &self,
        _request: WorkerProviderRequest,
        _cancellation: CancellationToken,
    ) -> Result<WorkerProviderResponse, String> {
        Err("worker provider calls are not supported".to_owned())
    }

    /// Executes one registered plugin slash command inside the worker process.
    ///
    /// - `command`: Exact command identifier advertised at startup.
    /// - `arguments`: Raw text after the command name.
    /// - `action`: Optional plugin UI action returned from a prior response.
    /// - `cancellation`: Shared host cancellation token.
    ///
    /// # Returns
    /// The structured command UI response.
    ///
    /// # Errors
    /// Returns a safe failure when command execution is unavailable.
    fn call_command(
        &self,
        _command: &str,
        _arguments: String,
        _action: Option<Value>,
        _cancellation: CancellationToken,
    ) -> Result<Value, String> {
        Err("worker slash commands are not supported".to_owned())
    }
}

/// Serves one plugin worker over a newline-delimited stream pair.
///
/// # Parameters
/// - `handler`: Worker tool implementation.
/// - `input`: Host command stream.
/// - `output`: Host response stream.
///
/// # Returns
/// `Ok(())` after an orderly shutdown or stream close.
///
/// # Errors
/// Returns an error when initialization or a response cannot be written.
pub fn serve_worker<R, W>(
    handler: Arc<dyn WorkerHandler>,
    input: R,
    output: W,
) -> std::io::Result<()>
where
    R: BufRead + Send + 'static,
    W: Write + Send + 'static,
{
    let writer: Arc<dyn WorkerFrameWriter> = Arc::new(SharedWriter(Arc::new(Mutex::new(output))));
    let host = Arc::new(WorkerHostClient::new(Arc::clone(&writer)));
    let (frames, receiver) = mpsc::channel();
    let reader = {
        let writer = Arc::clone(&writer);
        let host = Arc::clone(&host);
        thread::spawn(move || {
            for line in input.lines() {
                let frame = match line {
                    Ok(line) => WorkerFrame::decode(&line),
                    Err(_) => break,
                };
                match frame {
                    Ok(WorkerFrame::HostResult { id, result }) => host.resolve(id, Ok(result)),
                    Ok(WorkerFrame::HostError { id, error }) => host.resolve(id, Err(error)),
                    Ok(frame) => {
                        let shutdown = matches!(&frame, WorkerFrame::Shutdown);
                        if frames.send(frame).is_err() || shutdown {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = writer.write_frame(&WorkerFrame::ToolError { id: 0, error });
                    }
                }
            }
        })
    };
    handler
        .start(Arc::clone(&host))
        .map_err(std::io::Error::other)?;
    writer.write_frame(&WorkerFrame::hello_with_registrations(
        handler.tools(),
        handler.providers(),
        handler.commands(),
    ))?;
    let cancellations: Arc<Mutex<HashMap<u64, CancellationToken>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let active: Arc<Mutex<Vec<thread::JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));
    while let Ok(frame) = receiver.recv() {
        match frame {
            WorkerFrame::ToolCall {
                id,
                tool,
                arguments,
            } => {
                let cancellation = CancellationToken::new();
                if let Ok(mut cancellations) = cancellations.lock() {
                    cancellations.insert(id, cancellation.clone());
                }
                let handler = Arc::clone(&handler);
                let writer = Arc::clone(&writer);
                let cancellations = Arc::clone(&cancellations);
                let handle = thread::spawn(move || {
                    let response = handler
                        .call_tool(&tool, arguments, cancellation)
                        .map_or_else(
                            |error| WorkerFrame::ToolError { id, error },
                            |result| WorkerFrame::ToolResult { id, result },
                        );
                    if let Ok(mut cancellations) = cancellations.lock() {
                        cancellations.remove(&id);
                    }
                    let _ = writer.write_frame(&response);
                });
                if let Ok(mut active) = active.lock() {
                    active.retain(|handle| !handle.is_finished());
                    active.push(handle);
                }
            }
            WorkerFrame::ProviderCall { id, request } => {
                let cancellation = CancellationToken::new();
                if let Ok(mut cancellations) = cancellations.lock() {
                    cancellations.insert(id, cancellation.clone());
                }
                let handler = Arc::clone(&handler);
                let writer = Arc::clone(&writer);
                let cancellations = Arc::clone(&cancellations);
                let handle = thread::spawn(move || {
                    let response = handler.call_provider(request, cancellation).map_or_else(
                        |error| WorkerFrame::ProviderError { id, error },
                        |response| WorkerFrame::ProviderResult { id, response },
                    );
                    if let Ok(mut cancellations) = cancellations.lock() {
                        cancellations.remove(&id);
                    }
                    let _ = writer.write_frame(&response);
                });
                if let Ok(mut active) = active.lock() {
                    active.retain(|handle| !handle.is_finished());
                    active.push(handle);
                }
            }
            WorkerFrame::CommandCall {
                id,
                command,
                arguments,
                action,
            } => {
                let cancellation = CancellationToken::new();
                if let Ok(mut cancellations) = cancellations.lock() {
                    cancellations.insert(id, cancellation.clone());
                }
                let handler = Arc::clone(&handler);
                let writer = Arc::clone(&writer);
                let cancellations = Arc::clone(&cancellations);
                let handle = thread::spawn(move || {
                    let response = handler
                        .call_command(&command, arguments, action, cancellation)
                        .map_or_else(
                            |error| WorkerFrame::CommandError { id, error },
                            |result| WorkerFrame::CommandResult { id, result },
                        );
                    if let Ok(mut cancellations) = cancellations.lock() {
                        cancellations.remove(&id);
                    }
                    let _ = writer.write_frame(&response);
                });
                if let Ok(mut active) = active.lock() {
                    active.retain(|handle| !handle.is_finished());
                    active.push(handle);
                }
            }
            WorkerFrame::Cancel { id } => {
                if let Ok(cancellations) = cancellations.lock() {
                    if let Some(cancellation) = cancellations.get(&id) {
                        cancellation.cancel();
                    }
                }
            }
            WorkerFrame::Shutdown => break,
            _ => {}
        }
    }
    if let Ok(cancellations) = cancellations.lock() {
        for cancellation in cancellations.values() {
            cancellation.cancel();
        }
    }
    settle_tool_threads(&active, Duration::from_secs(2));
    let _ = reader.join();
    Ok(())
}

/// Waits for cancelled tool threads to finish without blocking forever.
///
/// # Parameters
/// - `active`: Tool threads spawned by the worker loop.
/// - `grace`: Maximum wait after cancellation.
///
/// # Returns
/// No value.
fn settle_tool_threads(active: &Arc<Mutex<Vec<thread::JoinHandle<()>>>>, grace: Duration) {
    let deadline = Instant::now() + grace;
    loop {
        let finished = match active.lock() {
            Ok(mut handles) => {
                let mut finished = Vec::new();
                let mut pending = Vec::new();
                for handle in handles.drain(..) {
                    if handle.is_finished() {
                        finished.push(handle);
                    } else {
                        pending.push(handle);
                    }
                }
                *handles = pending;
                finished
            }
            Err(_) => return,
        };
        for handle in finished {
            let _ = handle.join();
        }
        let settled = active.lock().is_ok_and(|handles| handles.is_empty());
        if settled || Instant::now() >= deadline {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    if active.lock().is_ok_and(|handles| !handles.is_empty()) {
        tracing::warn!(
            target: "rpi_extensions::worker",
            "plugin tool did not stop cooperatively; worker process exit is required"
        );
    }
}

/// Serves one worker over the process standard streams.
///
/// # Parameters
/// - `handler`: Worker tool implementation.
///
/// # Returns
/// `Ok(())` after orderly shutdown or standard-input close.
///
/// # Errors
/// Returns an error when a response frame cannot be written.
pub fn serve_worker_stdio(handler: Arc<dyn WorkerHandler>) -> std::io::Result<()> {
    serve_worker(
        handler,
        std::io::BufReader::new(std::io::stdin()),
        std::io::stdout(),
    )
}
