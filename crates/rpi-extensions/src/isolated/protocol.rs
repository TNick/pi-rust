//! Defines the versioned JSON-lines protocol for isolated plugin workers.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of the host/worker framing contract.
pub const WORKER_PROTOCOL: &str = "rpi-plugin-worker-v1";

/// One plugin tool advertised by a worker process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerToolSpec {
    /// Host-stamped tool identifier.
    pub name: String,
    /// Safe display description.
    pub description: String,
    /// Provider-facing JSON schema for arguments.
    pub parameters: Value,
}

/// One host/worker frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerFrame {
    /// Worker readiness with its registered tool set.
    Hello {
        /// Protocol version implemented by the worker.
        protocol: String,
        /// Tools the worker can execute.
        tools: Vec<WorkerToolSpec>,
    },
    /// Host requests one registered tool invocation.
    ToolCall {
        /// Correlation identity.
        id: u64,
        /// Registered tool identifier.
        tool: String,
        /// Validated tool arguments.
        arguments: Value,
    },
    /// Worker returns a completed tool result.
    ToolResult {
        /// Correlation identity.
        id: u64,
        /// Bounded JSON result.
        result: Value,
    },
    /// Worker reports a safe tool failure.
    ToolError {
        /// Correlation identity.
        id: u64,
        /// Bounded failure text.
        error: String,
    },
    /// Host cancels one in-flight tool invocation.
    Cancel {
        /// Correlation identity of the invocation to cancel.
        id: u64,
    },
    /// Worker requests a host-mediated capability.
    HostCall {
        /// Correlation identity.
        id: u64,
        /// Stable host capability identifier.
        capability: String,
        /// Capability arguments.
        arguments: Value,
    },
    /// Host returns a host-mediated capability result.
    HostResult {
        /// Correlation identity.
        id: u64,
        /// Bounded JSON result.
        result: Value,
    },
    /// Host reports a denied or failed host-mediated capability call.
    HostError {
        /// Correlation identity.
        id: u64,
        /// Bounded failure text.
        error: String,
    },
    /// Host requests orderly worker shutdown.
    Shutdown,
}

impl WorkerFrame {
    /// Builds a protocol-compliant worker readiness frame.
    ///
    /// # Parameters
    /// - `tools`: Tools the worker can execute.
    ///
    /// # Returns
    /// A `hello` frame stamped with the current protocol version.
    pub fn hello(tools: Vec<WorkerToolSpec>) -> Self {
        Self::Hello {
            protocol: WORKER_PROTOCOL.to_owned(),
            tools,
        }
    }

    /// Serializes one frame without its trailing newline.
    ///
    /// # Returns
    /// UTF-8 JSON bytes.
    ///
    /// # Errors
    /// Returns a serialization error when the frame cannot be encoded.
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Parses one newline-delimited frame.
    ///
    /// # Parameters
    /// - `line`: One JSON line without its trailing newline.
    ///
    /// # Returns
    /// The decoded frame.
    ///
    /// # Errors
    /// Returns a stable parse error without echoing the malformed payload.
    pub fn decode(line: &str) -> Result<Self, String> {
        serde_json::from_str(line.trim()).map_err(|_| "invalid plugin-worker frame".to_owned())
    }

    /// Validates a worker `hello` protocol version.
    ///
    /// # Returns
    /// `Ok(())` when this frame is a compatible `hello`.
    ///
    /// # Errors
    /// Returns a safe protocol error for another frame or version.
    pub fn validate_hello(&self) -> Result<Vec<WorkerToolSpec>, String> {
        match self {
            Self::Hello { protocol, tools } if protocol == WORKER_PROTOCOL => Ok(tools.clone()),
            Self::Hello { .. } => Err("plugin worker protocol version mismatch".to_owned()),
            _ => Err("plugin worker did not start with a hello frame".to_owned()),
        }
    }
}
