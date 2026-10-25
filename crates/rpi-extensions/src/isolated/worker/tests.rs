//! Verifies isolated provider requests use the versioned worker protocol.

use std::io::{self, Cursor, Write};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::isolated::protocol::{
    WorkerCommandSpec, WorkerFrame, WorkerProviderOptions, WorkerProviderOutcome,
    WorkerProviderRequest, WorkerProviderResponse, WorkerProviderSpec,
};
use crate::isolated::worker::{WorkerHandler, serve_worker};

/// Captures provider calls made by the worker service test.
struct ProviderFixture {
    /// Provider credential observed by the host callback fixture.
    observed_secret: Arc<Mutex<Option<String>>>,
}

impl WorkerHandler for ProviderFixture {
    /// Advertises no tools in this provider-only fixture.
    fn tools(&self) -> Vec<super::super::protocol::WorkerToolSpec> {
        Vec::new()
    }

    /// Advertises one provider model through startup readiness.
    fn providers(&self) -> Vec<WorkerProviderSpec> {
        vec![WorkerProviderSpec {
            id: "fixture-provider".to_owned(),
            models: vec![rpi_ai::Model::new(
                "fixture-model",
                "Fixture Model",
                rpi_ai::Api::Faux,
                "fixture-provider",
                "https://fixture.example.test/v1",
            )],
        }]
    }

    /// Advertises one command for the worker protocol test.
    fn commands(&self) -> Vec<WorkerCommandSpec> {
        vec![WorkerCommandSpec {
            name: "fixture-command".to_owned(),
            description: "Returns a fixture message.".to_owned(),
        }]
    }

    /// Captures the protected credential before returning a fixed response.
    ///
    /// # Parameters
    /// - `request`: Host-authorized provider request.
    /// - `_cancellation`: Worker request cancellation token.
    ///
    /// # Returns
    /// A fixed assistant response with a successful stop outcome.
    ///
    /// # Errors
    /// Returns an error if the fixture's capture lock is poisoned.
    fn call_provider(
        &self,
        request: WorkerProviderRequest,
        _cancellation: CancellationToken,
    ) -> Result<WorkerProviderResponse, String> {
        *self
            .observed_secret
            .lock()
            .map_err(|_| "fixture lock is poisoned".to_owned())? = request.secret;
        Ok(WorkerProviderResponse {
            message: rpi_ai::AssistantMessage::terminal(
                rpi_ai::Api::Faux,
                "fixture-provider",
                "fixture-model",
                rpi_ai::StopReason::Stop,
                "worker provider completed",
                1,
            ),
            outcome: WorkerProviderOutcome::Stop,
        })
    }

    /// Rejects tool calls because this fixture advertises no tools.
    ///
    /// # Parameters
    /// - `_tool`: Requested tool name.
    /// - `_arguments`: Requested tool arguments.
    /// - `_cancellation`: Worker request cancellation token.
    ///
    /// # Returns
    /// Never returns a tool result.
    ///
    /// # Errors
    /// Always returns a fixture error.
    fn call_tool(
        &self,
        _tool: &str,
        _arguments: Value,
        _cancellation: CancellationToken,
    ) -> Result<Value, String> {
        Err("fixture does not support tools".to_owned())
    }

    /// Returns an echo response for one registered fixture command.
    ///
    /// - `command`: Advertised command name.
    /// - `arguments`: Raw arguments supplied by the host.
    /// - `_cancellation`: Worker request cancellation token.
    ///
    /// # Returns
    /// A safe message response containing the supplied arguments.
    ///
    /// # Errors
    /// Returns an error when the requested command name is not recognized.
    fn call_command(
        &self,
        command: &str,
        arguments: String,
        _cancellation: CancellationToken,
    ) -> Result<Value, String> {
        if command != "fixture-command" {
            return Err("unknown fixture command".to_owned());
        }
        Ok(serde_json::json!({"kind": "message", "text": arguments}))
    }
}

/// Shares captured worker output with the test after the service exits.
struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuffer {
    /// Appends worker bytes to the shared test buffer.
    ///
    /// # Parameters
    /// - `bytes`: Bytes emitted by the worker.
    ///
    /// # Returns
    /// Number of bytes appended.
    ///
    /// # Errors
    /// Returns an error when the fixture buffer lock is poisoned.
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut buffer = self
            .0
            .lock()
            .map_err(|_| io::Error::other("fixture output lock is poisoned"))?;
        buffer.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    /// Flushes the in-memory worker output buffer.
    ///
    /// # Returns
    /// `Ok(())` because no operating-system buffer is involved.
    ///
    /// # Errors
    /// This fixture does not produce a flush error.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Proves provider catalog advertisement and protected calls round-trip.
#[test]
fn provider_registration_and_request_round_trip_through_worker_protocol() {
    let request = WorkerProviderRequest {
        provider: "fixture-provider".to_owned(),
        model: "fixture-model".to_owned(),
        context: rpi_ai::Context::default(),
        options: WorkerProviderOptions::default(),
        secret: Some("secret-material".to_owned()),
        bearer: true,
    };
    assert!(!format!("{request:?}").contains("secret-material"));
    let request_frame = WorkerFrame::ProviderCall { id: 7, request };
    let mut input = request_frame.encode().expect("provider frame encoding");
    input.push(b'\n');
    input.extend_from_slice(
        &WorkerFrame::CommandCall {
            id: 8,
            command: "fixture-command".to_owned(),
            arguments: "hello from host".to_owned(),
        }
        .encode()
        .expect("command frame encoding"),
    );
    input.push(b'\n');
    input.extend_from_slice(
        &WorkerFrame::Shutdown
            .encode()
            .expect("shutdown frame encoding"),
    );
    input.push(b'\n');

    let observed_secret = Arc::new(Mutex::new(None));
    let handler = Arc::new(ProviderFixture {
        observed_secret: Arc::clone(&observed_secret),
    });
    let output = Arc::new(Mutex::new(Vec::new()));
    serve_worker(
        handler,
        Cursor::new(input),
        SharedBuffer(Arc::clone(&output)),
    )
    .expect("serve worker request");

    assert_eq!(
        observed_secret.lock().expect("captured secret").as_deref(),
        Some("secret-material")
    );
    let output = output.lock().expect("worker output");
    let frames = String::from_utf8(output.clone())
        .expect("worker output is utf-8")
        .lines()
        .map(WorkerFrame::decode)
        .collect::<Result<Vec<_>, _>>()
        .expect("worker output frames decode");
    let registrations = frames[0]
        .validate_hello_registrations()
        .expect("provider catalog hello");
    assert!(registrations.tools.is_empty());
    assert_eq!(registrations.commands[0].name, "fixture-command");
    assert_eq!(registrations.providers[0].id, "fixture-provider");
    assert_eq!(registrations.providers[0].models[0].id, "fixture-model");
    assert!(matches!(
        frames[1],
        WorkerFrame::ProviderResult {
            id: 7,
            response: WorkerProviderResponse {
                outcome: WorkerProviderOutcome::Stop,
                ..
            },
        }
    ));
    assert!(matches!(
        frames.iter().find(|frame| matches!(frame, WorkerFrame::CommandResult { id: 8, .. })),
        Some(WorkerFrame::CommandResult { id: 8, result })
            if result["text"] == "hello from host"
    ));
    assert!(!String::from_utf8_lossy(&output).contains("secret-material"));
}
