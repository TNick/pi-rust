//! Defines the versioned JSON-lines protocol for isolated plugin workers.

use rpi_ai::{AssistantMessage, Context, Model, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of the host/worker framing contract.
pub const WORKER_PROTOCOL: &str = "rpi-plugin-worker-v2";

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

/// One plugin slash command advertised by an isolated worker.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerCommandSpec {
    /// Host-stamped command identifier without a leading slash.
    pub name: String,
    /// Safe user-facing command description.
    pub description: String,
}

/// One provider and its host-safe model catalog advertised by a worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerProviderSpec {
    /// Host-selected provider identity.
    pub id: String,
    /// Credential-free models served by the provider.
    pub models: Vec<Model>,
}

/// Registrations returned by a worker during startup.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerRegistrations {
    /// Worker tools available for host authorization.
    pub tools: Vec<WorkerToolSpec>,
    /// Slash commands available for host-mediated invocation.
    #[serde(default)]
    pub commands: Vec<WorkerCommandSpec>,
    /// Provider models available for host catalog publication.
    pub providers: Vec<WorkerProviderSpec>,
}

/// Non-secret request settings accepted by an isolated provider call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerProviderOptions {
    /// Request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum number of retry attempts.
    pub max_retries: Option<u32>,
    /// Reasoning level requested by the host.
    pub reasoning: Option<ThinkingLevel>,
    /// Requested output token limit.
    pub max_tokens: Option<u64>,
    /// Requested sampling temperature.
    pub temperature: Option<f64>,
}

/// One request sent to a provider registered by a worker plugin.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerProviderRequest {
    /// Exact provider identity advertised during worker startup.
    pub provider: String,
    /// Exact model identity advertised for that provider.
    pub model: String,
    /// Provider-neutral request context.
    pub context: Context,
    /// Non-secret request options.
    pub options: WorkerProviderOptions,
    /// Request-only host credential carried over the private worker pipe.
    pub secret: Option<String>,
    /// Whether the request credential uses bearer authorization.
    pub bearer: bool,
}

/// Terminal provider response crossing the isolated worker boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerProviderResponse {
    /// Terminal assistant message produced by the extension provider.
    pub message: AssistantMessage,
    /// Terminal category preserved for host authentication and cancellation logic.
    pub outcome: WorkerProviderOutcome,
}

/// Safe terminal category reported for a worker provider call.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderOutcome {
    /// Provider returned a completed response.
    Stop,
    /// Provider completed at the requested output limit.
    Length,
    /// Provider completed with a tool call.
    ToolUse,
    /// Provider returned a deferred response.
    Deferred,
    /// Provider explicitly rejected request authentication.
    Authentication,
    /// Provider failed for another reason.
    Error,
    /// Request was cancelled.
    Aborted,
}

impl std::fmt::Debug for WorkerProviderRequest {
    /// Formats a provider request without revealing its credential.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerProviderRequest")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("context", &"[provider context]")
            .field("options", &self.options)
            .field("secret", &self.secret.as_ref().map(|_| "[redacted]"))
            .field("bearer", &self.bearer)
            .finish()
    }
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
        /// Providers and models the worker can execute.
        #[serde(default)]
        providers: Vec<WorkerProviderSpec>,
        /// Slash commands the worker can execute.
        #[serde(default)]
        commands: Vec<WorkerCommandSpec>,
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
    /// Host requests one registered slash command invocation.
    CommandCall {
        /// Correlation identity.
        id: u64,
        /// Registered command identifier.
        command: String,
        /// Raw command arguments supplied after the command name.
        arguments: String,
        /// Optional structured action returned by a plugin command UI.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<Value>,
    },
    /// Worker returns a command's structured UI response.
    CommandResult {
        /// Correlation identity.
        id: u64,
        /// Structured UI result from the plugin command callback.
        result: Value,
    },
    /// Worker reports a safe command failure.
    CommandError {
        /// Correlation identity.
        id: u64,
        /// Bounded failure text.
        error: String,
    },
    /// Host requests one provider turn from the worker.
    ProviderCall {
        /// Correlation identity.
        id: u64,
        /// Provider request authorized by the host.
        request: WorkerProviderRequest,
    },
    /// Worker returns a completed provider turn.
    ProviderResult {
        /// Correlation identity.
        id: u64,
        /// Terminal assistant response.
        response: WorkerProviderResponse,
    },
    /// Worker reports a safe provider failure.
    ProviderError {
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
            providers: Vec::new(),
            commands: Vec::new(),
        }
    }

    /// Builds readiness with tools and isolated provider registrations.
    ///
    /// # Parameters
    /// - `tools`: Worker tools the host may authorize.
    /// - `providers`: Credential-free provider catalog advertised by the worker.
    ///
    /// # Returns
    /// A `hello` frame stamped with the current protocol version.
    pub fn hello_with_providers(
        tools: Vec<WorkerToolSpec>,
        providers: Vec<WorkerProviderSpec>,
    ) -> Self {
        Self::Hello {
            protocol: WORKER_PROTOCOL.to_owned(),
            tools,
            providers,
            commands: Vec::new(),
        }
    }

    /// Builds readiness with tools, providers, and slash commands.
    ///
    /// - `tools`: Worker tools the host may authorize.
    /// - `providers`: Credential-free provider catalog advertised by the worker.
    /// - `commands`: Slash commands the host may expose and invoke.
    ///
    /// # Returns
    /// A `hello` frame stamped with the current protocol version.
    pub fn hello_with_registrations(
        tools: Vec<WorkerToolSpec>,
        providers: Vec<WorkerProviderSpec>,
        commands: Vec<WorkerCommandSpec>,
    ) -> Self {
        Self::Hello {
            protocol: WORKER_PROTOCOL.to_owned(),
            tools,
            providers,
            commands,
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
            Self::Hello {
                protocol, tools, ..
            } if protocol == WORKER_PROTOCOL => Ok(tools.clone()),
            Self::Hello { .. } => Err("plugin worker protocol version mismatch".to_owned()),
            _ => Err("plugin worker did not start with a hello frame".to_owned()),
        }
    }

    /// Validates and returns all worker registrations from a readiness frame.
    ///
    /// # Returns
    /// The worker's tool and provider registrations.
    ///
    /// # Errors
    /// Returns a protocol error for a non-hello frame or incompatible version.
    pub fn validate_hello_registrations(&self) -> Result<WorkerRegistrations, String> {
        match self {
            Self::Hello {
                protocol,
                tools,
                providers,
                commands,
            } if protocol == WORKER_PROTOCOL => Ok(WorkerRegistrations {
                tools: tools.clone(),
                providers: providers.clone(),
                commands: commands.clone(),
            }),
            Self::Hello { .. } => Err("plugin worker protocol version mismatch".to_owned()),
            _ => Err("plugin worker did not start with a hello frame".to_owned()),
        }
    }
}
