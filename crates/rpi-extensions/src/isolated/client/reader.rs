//! Routes worker responses, host-capability calls, and connection closure.

use std::io::{BufRead, BufReader, Write};
use std::process::ChildStdin;
use std::sync::{mpsc, Arc, Mutex};

use serde_json::Value;

use super::super::protocol::{WorkerFrame, WorkerToolSpec};
use super::{Pending, WorkerError, WorkerHostHandler};

/// Reads frames until the worker exits, routing responses and host calls.
///
/// # Parameters
/// - `output`: Worker stdout.
/// - `input`: Shared worker stdin for host responses.
/// - `pending`: Outstanding request senders.
/// - `closed`: Shared connection failure state.
/// - `hello`: One-shot readiness sender.
/// - `host_handler`: Optional host capability policy.
pub(super) fn read_worker(
    output: impl std::io::Read + Send + 'static,
    input: Arc<Mutex<ChildStdin>>,
    pending: Pending,
    closed: Arc<Mutex<Option<String>>>,
    hello: mpsc::Sender<Result<Vec<WorkerToolSpec>, WorkerError>>,
    host_handler: Option<Arc<dyn WorkerHostHandler>>,
) {
    let mut hello = Some(hello);
    for line in BufReader::new(output).lines().map_while(Result::ok) {
        let frame = match WorkerFrame::decode(&line) {
            Ok(frame) => frame,
            Err(error) => {
                close(&closed, &pending, &error);
                if let Some(sender) = hello.take() {
                    let _ = sender.send(Err(WorkerError::Protocol(error)));
                }
                return;
            }
        };
        if matches!(&frame, WorkerFrame::Hello { .. }) {
            if let Some(sender) = hello.take() {
                let result = frame.validate_hello().map_err(WorkerError::Protocol);
                let _ = sender.send(result);
            }
            continue;
        }
        match frame {
            WorkerFrame::ToolResult { id, result } => {
                resolve(&pending, id, Ok(result));
            }
            WorkerFrame::ToolError { id, error } => {
                resolve(&pending, id, Err(error));
            }
            WorkerFrame::HostCall {
                id,
                capability,
                arguments,
            } => {
                let response = match host_handler.as_ref() {
                    Some(handler) => handler.call(&capability, &arguments),
                    None => Err("host capability is not available".to_owned()),
                };
                let frame = match response {
                    Ok(result) => WorkerFrame::HostResult { id, result },
                    Err(error) => WorkerFrame::HostError { id, error },
                };
                if let Ok(bytes) = frame.encode() {
                    if let Ok(mut input) = input.lock() {
                        let _ = input.write_all(&bytes).and_then(|()| {
                            input.write_all(b"\n")?;
                            input.flush()
                        });
                    }
                }
            }
            WorkerFrame::Shutdown => break,
            _ => {}
        }
    }
    close(&closed, &pending, "plugin worker exited unexpectedly");
    if let Some(sender) = hello.take() {
        let _ = sender.send(Err(WorkerError::Closed(
            "plugin worker exited before readiness".to_owned(),
        )));
    }
}

/// Resolves one pending request.
///
/// # Parameters
/// - `pending`: Outstanding request map.
/// - `id`: Correlation identity.
/// - `result`: Response value.
fn resolve(pending: &Pending, id: u64, result: Result<Value, String>) {
    if let Ok(mut pending) = pending.lock() {
        if let Some(sender) = pending.remove(&id) {
            let _ = sender.send(result);
        }
    }
}

/// Marks the connection closed and fails every outstanding request.
///
/// # Parameters
/// - `closed`: Shared connection state.
/// - `pending`: Outstanding request map.
/// - `message`: Safe connection failure.
fn close(closed: &Arc<Mutex<Option<String>>>, pending: &Pending, message: &str) {
    if let Ok(mut state) = closed.lock() {
        *state = Some(message.to_owned());
    }
    if let Ok(mut pending) = pending.lock() {
        for (_, sender) in pending.drain() {
            let _ = sender.send(Err(message.to_owned()));
        }
    }
}
