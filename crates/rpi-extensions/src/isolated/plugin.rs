//! Loads and executes native plugins only inside an isolated worker process.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rpi_agent::agent_tool::AgentTool;
use rpi_agent::types::{AgentToolResult, TextContentOrImage};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::actions::{ActionBridge, RuntimeActionHost};
use crate::loader::{PluginKeepalive, load_session_mixed};
use crate::registry::ExtensionTool;
use crate::tool::PluginToolAdapter;
use crate::{NullDiagnostics, PluginDiagnostics};

use super::protocol::{
    WorkerCommandSpec, WorkerProviderOptions, WorkerProviderOutcome, WorkerProviderRequest,
    WorkerProviderResponse, WorkerProviderSpec, WorkerToolSpec,
};
use super::worker::{WorkerHandler, WorkerHostClient, serve_worker_stdio};

#[cfg(test)]
#[path = "plugin/tests.rs"]
mod tests;

/// Native plugin set loaded inside a worker process.
///
/// # Public members
/// - [`PluginWorker::new`]: Selects the plugin roots without loading them.
/// - [`PluginWorker::loaded_paths`]: Loaded plugin paths after worker start.
///
/// # Private members
/// - `directories`: Authorized plugin directories.
/// - `files`: Authorized explicit plugin files.
/// - `diagnostics`: Worker-local diagnostics sink.
/// - `loaded`: Plugins, runtime bridge, and keepalive after host setup.
pub struct PluginWorker {
    /// Authorized plugin directories.
    directories: Vec<PathBuf>,
    /// Authorized explicit plugin files.
    files: Vec<PathBuf>,
    /// Worker-local diagnostics sink.
    diagnostics: Arc<dyn PluginDiagnostics>,
    /// Loaded plugin state after the host bridge is available.
    loaded: Mutex<Option<LoadedPlugins>>,
}

/// Runtime and plugin state loaded after the host bridge exists.
struct LoadedPlugins {
    /// Library keepalive owned by this worker only.
    keepalive: Arc<PluginKeepalive>,
    /// Registered tools whose callbacks live inside the worker's libraries.
    tools: Vec<ExtensionTool>,
    /// Registered slash command callbacks that remain inside the worker.
    commands: Vec<crate::registry::RegisteredCommand>,
    /// Provider implementations whose callbacks remain inside the worker.
    providers: Vec<Arc<dyn rpi_ai::Provider>>,
    /// Loaded plugin paths for diagnostics.
    loaded_paths: Vec<PathBuf>,
    /// Runtime serving plugin runtime-action dispatch.
    runtime: tokio::runtime::Runtime,
    /// Action bridge retained with the loaded plugin set.
    _bridge: Arc<ActionBridge>,
}

impl PluginWorker {
    /// Selects plugin roots for loading when the worker host is attached.
    ///
    /// # Parameters
    /// - `directories`: Plugin directories selected by the host.
    /// - `files`: Explicit plugin files selected by the host.
    /// - `diagnostics`: Worker-local diagnostics sink.
    ///
    /// # Returns
    /// A worker ready to load plugins in [`WorkerHandler::start`].
    pub fn new(
        directories: &[PathBuf],
        files: &[PathBuf],
        diagnostics: Arc<dyn PluginDiagnostics>,
    ) -> Self {
        Self {
            directories: directories.to_vec(),
            files: files.to_vec(),
            diagnostics,
            loaded: Mutex::new(None),
        }
    }

    /// Returns the loaded plugin paths for diagnostics.
    pub fn loaded_paths(&self) -> Vec<PathBuf> {
        self.loaded
            .lock()
            .ok()
            .and_then(|loaded| loaded.as_ref().map(|loaded| loaded.loaded_paths.clone()))
            .unwrap_or_default()
    }
}

impl WorkerHandler for PluginWorker {
    /// Projects registered plugin tools into the worker protocol.
    fn tools(&self) -> Vec<WorkerToolSpec> {
        let Ok(loaded) = self.loaded.lock() else {
            return Vec::new();
        };
        loaded
            .as_ref()
            .map(|loaded| {
                loaded
                    .tools
                    .iter()
                    .map(|tool| WorkerToolSpec {
                        name: tool.tool.name.clone(),
                        description: tool.tool.description.clone(),
                        parameters: serde_json::to_value(&tool.tool.parameters)
                            .unwrap_or(Value::Null),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Projects provider manifests into a credential-free worker catalog.
    fn providers(&self) -> Vec<WorkerProviderSpec> {
        let Ok(loaded) = self.loaded.lock() else {
            return Vec::new();
        };
        loaded
            .as_ref()
            .map(|loaded| {
                loaded
                    .providers
                    .iter()
                    .map(|provider| WorkerProviderSpec {
                        id: provider.id().to_owned(),
                        models: provider
                            .models()
                            .iter()
                            .cloned()
                            .map(|mut model| {
                                model.provider = provider.id().to_owned();
                                model.headers = None;
                                model
                            })
                            .collect(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Projects registered slash commands into the worker protocol.
    fn commands(&self) -> Vec<WorkerCommandSpec> {
        let Ok(loaded) = self.loaded.lock() else {
            return Vec::new();
        };
        loaded
            .as_ref()
            .map(|loaded| {
                loaded
                    .commands
                    .iter()
                    .map(|command| WorkerCommandSpec {
                        name: command.name.trim_start_matches('/').to_owned(),
                        description: command.description.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Loads plugins with a bridge that relays runtime actions to the host.
    ///
    /// # Parameters
    /// - `host`: Worker-side host capability client.
    ///
    /// # Errors
    /// Returns a safe failure when no plugin or runtime can be prepared.
    fn start(&self, host: Arc<WorkerHostClient>) -> Result<(), String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        let action_host = Arc::new(WorkerActionHost {
            client: Arc::clone(&host),
        });
        let bridge =
            ActionBridge::new(runtime.handle().clone(), action_host).with_remote_ui_components();
        let session = load_session_mixed(
            &self.directories,
            &self.files,
            Arc::clone(&self.diagnostics),
            Some(Arc::clone(&bridge)),
        );
        if session.is_empty() {
            return Err("no native plugin loaded in the worker".to_owned());
        }
        let keepalive = session.keepalive();
        let tools = session
            .snapshot()
            .map(|snapshot| snapshot.tools().to_vec())
            .unwrap_or_default();
        let commands = session
            .snapshot()
            .map(|snapshot| snapshot.commands().to_vec())
            .unwrap_or_default();
        let providers = crate::PluggableProvider::from_session(&session, runtime.handle().clone());
        if tools.is_empty() && commands.is_empty() && providers.is_empty() {
            return Err("no plugin tool or provider registered in the worker".to_owned());
        }
        let loaded_paths = session.loaded_paths().to_vec();
        drop(session);
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| "worker plugin state is poisoned".to_owned())?;
        *loaded = Some(LoadedPlugins {
            keepalive,
            tools,
            commands,
            providers,
            loaded_paths,
            runtime,
            _bridge: bridge,
        });
        Ok(())
    }

    /// Executes one registered plugin tool inside the worker process.
    ///
    /// # Errors
    /// Returns a safe error when the tool is unknown or the plugin call fails.
    fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<Value, String> {
        if cancellation.is_cancelled() {
            return Err("plugin tool cancelled".to_owned());
        }
        let loaded = self
            .loaded
            .lock()
            .map_err(|_| "worker plugin state is poisoned".to_owned())?;
        let loaded = loaded
            .as_ref()
            .ok_or_else(|| "worker plugins are not loaded".to_owned())?;
        let registered = loaded
            .tools
            .iter()
            .find(|tool| tool.tool.name == tool_name)
            .ok_or_else(|| "plugin tool is not registered".to_owned())?;
        let adapter = PluginToolAdapter::from_registered(registered, Arc::clone(&loaded.keepalive))
            .with_pre_execute_authorizer(|_, _| true);
        let result = loaded.runtime.block_on(adapter.execute(
            "isolated-worker-call",
            arguments,
            cancellation.clone(),
            Arc::new(|_| {}),
        ));
        if cancellation.is_cancelled() {
            return Err("plugin tool cancelled".to_owned());
        }
        result
            .map(|result| result_to_json(&result))
            .map_err(|error| error.to_string())
    }

    /// Executes one extension provider request inside the isolated process.
    ///
    /// # Parameters
    /// - `request`: Host-authorized request bound to an advertised model.
    /// - `cancellation`: Host cancellation token.
    ///
    /// # Returns
    /// The provider's terminal assistant response.
    ///
    /// # Errors
    /// Returns a safe error when the provider is unknown, the request is
    /// cancelled, or the provider fails.
    fn call_provider(
        &self,
        request: WorkerProviderRequest,
        cancellation: CancellationToken,
    ) -> Result<WorkerProviderResponse, String> {
        if cancellation.is_cancelled() {
            return Err("plugin provider request cancelled".to_owned());
        }
        let loaded = self
            .loaded
            .lock()
            .map_err(|_| "worker plugin state is poisoned".to_owned())?;
        let loaded = loaded
            .as_ref()
            .ok_or_else(|| "worker plugins are not loaded".to_owned())?;
        let provider = loaded
            .providers
            .iter()
            .find(|provider| provider.id() == request.provider)
            .ok_or_else(|| "plugin provider is not registered".to_owned())?;
        let model = provider
            .models()
            .iter()
            .find(|model| model.id == request.model)
            .ok_or_else(|| "plugin provider model is not registered".to_owned())?;
        let mut options = provider_options(request.options);
        options.signal = cancellation.clone();
        let applied_settings = Arc::new(Mutex::new(None));
        let report_slot = Arc::clone(&applied_settings);
        let callback: rpi_ai::AppliedRequestSettingsCallback = Arc::new(move |report| {
            *report_slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(report);
        });
        let result = if let Some(secret) = request.secret {
            let secret = rpi_ai::RequestSecret::new(Arc::new(WorkerSecret(secret)));
            loaded.runtime.block_on(async {
                drain_provider_stream(
                    provider
                        .stream_simple_with_request_secret_and_applied_settings(
                            model,
                            &request.context,
                            &options,
                            &secret,
                            request.bearer,
                            callback,
                        )
                        .await,
                    cancellation,
                    applied_settings,
                )
                .await
            })
        } else {
            if request.bearer {
                return Err("bearer authorization requires a host credential".to_owned());
            }
            loaded.runtime.block_on(async {
                drain_provider_stream(
                    provider
                        .stream_simple_with_applied_settings(
                            model,
                            &request.context,
                            &options,
                            callback,
                        )
                        .await,
                    cancellation,
                    applied_settings,
                )
                .await
            })
        };
        result
    }

    /// Invokes a registered slash command callback inside the worker.
    ///
    /// - `command_name`: Exact normalized command identifier.
    /// - `arguments`: Raw command arguments from the host.
    /// - `action`: Optional structured UI action from a prior response.
    /// - `cancellation`: Shared host cancellation token.
    ///
    /// # Returns
    /// The plugin's structured command UI response.
    ///
    /// # Errors
    /// Returns a safe error when the command is unknown, cancelled, panics, or
    /// returns invalid JSON.
    fn call_command(
        &self,
        command_name: &str,
        arguments: String,
        action: Option<Value>,
        cancellation: CancellationToken,
    ) -> Result<Value, String> {
        if cancellation.is_cancelled() {
            return Err("plugin command cancelled".to_owned());
        }
        let loaded = self
            .loaded
            .lock()
            .map_err(|_| "worker plugin state is poisoned".to_owned())?;
        let loaded = loaded
            .as_ref()
            .ok_or_else(|| "worker plugins are not loaded".to_owned())?;
        let command = loaded
            .commands
            .iter()
            .find(|command| command.name.trim_start_matches('/') == command_name)
            .ok_or_else(|| "plugin command is not registered".to_owned())?;
        let mut input = serde_json::json!({
            "args": arguments,
            "command": format!("/{command_name}"),
        });
        if let Some(action) = action {
            input["action"] = action;
        }
        let input = input.to_string();
        let mut output = rpi_plugin_sdk::StbString::empty();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let status = (command.handler)(
                rpi_plugin_sdk::StbStringRef::from_str(&input),
                &mut output,
                command.user_data,
            );
            // SAFETY: the registered callback transfers output to its producer.
            let output = unsafe {
                rpi_plugin_sdk::callbacks::OwnedString::from_raw(
                    output,
                    Some(crate::host_free_string),
                )
            };
            if status != 0 {
                return Err("plugin command callback failed".to_owned());
            }
            serde_json::from_str(&output.to_string_lossy())
                .map_err(|_| "plugin command returned invalid JSON".to_owned())
        }))
        .map_err(|_| "plugin command callback panicked".to_owned())?;
        result
    }
}

/// Owns a request credential only for its worker-local provider callback.
struct WorkerSecret(String);

impl rpi_ai::SecretMaterial for WorkerSecret {
    /// Lends the secret to one synchronous request builder.
    fn with_secret(&self, consumer: &mut dyn FnMut(&str)) {
        consumer(&self.0);
    }
}

/// Converts public IPC options into upstream provider options.
///
/// # Parameters
/// - `options`: Non-secret request settings received over the private worker pipe.
///
/// # Returns
/// Provider options without an API key, headers, or ambient metadata.
fn provider_options(options: WorkerProviderOptions) -> rpi_ai::SimpleStreamOptions {
    rpi_ai::SimpleStreamOptions {
        timeout: options.timeout_ms.map(std::time::Duration::from_millis),
        max_retries: options.max_retries,
        reasoning: options.reasoning,
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        ..rpi_ai::SimpleStreamOptions::default()
    }
}

/// Drains the worker-local one-shot extension provider stream.
///
/// # Parameters
/// - `stream`: Provider stream to drain.
/// - `cancellation`: Host cancellation token forwarded to the provider.
/// - `applied_settings`: Shared slot populated by the provider's safe callback.
///
/// # Returns
/// Its terminal assistant response.
///
/// # Errors
/// Returns an error when the stream ends without a terminal event or is cancelled.
async fn drain_provider_stream(
    mut stream: rpi_ai::AssistantMessageEventStream,
    cancellation: CancellationToken,
    applied_settings: Arc<Mutex<Option<rpi_ai::AppliedRequestSettings>>>,
) -> Result<WorkerProviderResponse, String> {
    while let Some(event) = tokio::select! {
        () = cancellation.cancelled() => return Err("plugin provider request cancelled".to_owned()),
        event = stream.next() => event,
    } {
        match event {
            rpi_ai::AssistantMessageEvent::Done { reason, message } => {
                let outcome = match reason {
                    rpi_ai::types::DoneReason::Stop => WorkerProviderOutcome::Stop,
                    rpi_ai::types::DoneReason::Length => WorkerProviderOutcome::Length,
                    rpi_ai::types::DoneReason::ToolUse => WorkerProviderOutcome::ToolUse,
                    rpi_ai::types::DoneReason::Deferred => WorkerProviderOutcome::Deferred,
                };
                let applied_settings = applied_settings
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                return Ok(WorkerProviderResponse {
                    message,
                    outcome,
                    applied_settings,
                });
            }
            rpi_ai::AssistantMessageEvent::Error { reason, error } => {
                let outcome = match reason {
                    rpi_ai::types::ErrorReason::Aborted => WorkerProviderOutcome::Aborted,
                    rpi_ai::types::ErrorReason::Authentication => {
                        WorkerProviderOutcome::Authentication
                    }
                    rpi_ai::types::ErrorReason::Error => WorkerProviderOutcome::Error,
                };
                return Ok(WorkerProviderResponse {
                    message: error,
                    outcome,
                    applied_settings: applied_settings
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take(),
                });
            }
            _ => {}
        }
    }
    Err("plugin provider stream ended without a result".to_owned())
}

/// Relays plugin runtime actions to the host over the worker protocol.
struct WorkerActionHost {
    /// Host capability client owned by the worker loop.
    client: Arc<WorkerHostClient>,
}

impl WorkerActionHost {
    /// Sends one capability request to the host.
    ///
    /// # Parameters
    /// - `capability`: Stable runtime-action capability name.
    /// - `arguments`: Runtime-action arguments.
    ///
    /// # Returns
    /// The host result.
    ///
    /// # Errors
    /// Returns a host denial or timeout.
    fn relay(&self, capability: &str, arguments: Value) -> Result<Value, String> {
        self.client.call(capability, arguments)
    }
}

#[async_trait::async_trait]
impl RuntimeActionHost for WorkerActionHost {
    /// Relays plugin-owned component updates to the isolated worker host.
    ///
    /// # Parameters
    /// - `owner`: Worker-local plugin identity.
    /// - `args`: Component lifecycle operation.
    ///
    /// # Returns
    /// The host mailbox result.
    ///
    /// # Errors
    /// Returns a host denial or timeout.
    async fn ui_component(&self, owner: u64, args: Value) -> Result<Value, String> {
        self.relay("ui_component", json!({"owner": owner, "arguments": args}))
    }

    /// Relays `send_message` to the host.
    async fn send_message(&self, args: Value) -> Result<Value, String> {
        self.relay("send_message", args)
    }

    /// Relays `send_user_message` to the host.
    async fn send_user_message(&self, args: Value) -> Result<Value, String> {
        self.relay("send_user_message", args)
    }

    /// Relays `append_entry` to the host.
    async fn append_entry(&self, args: Value) -> Result<Value, String> {
        self.relay("append_entry", args)
    }

    /// Relays `set_session_name` to the host.
    async fn set_session_name(&self, args: Value) -> Result<Value, String> {
        self.relay("set_session_name", args)
    }

    /// Relays `get_active_tools` to the host.
    async fn get_active_tools(&self, args: Value) -> Result<Value, String> {
        self.relay("get_active_tools", args)
    }

    /// Relays `set_active_tools` to the host.
    async fn set_active_tools(&self, args: Value) -> Result<Value, String> {
        self.relay("set_active_tools", args)
    }

    /// Relays `set_model` to the host.
    async fn set_model(&self, args: Value) -> Result<Value, String> {
        self.relay("set_model", args)
    }

    /// Relays `get_thinking_level` to the host.
    async fn get_thinking_level(&self, args: Value) -> Result<Value, String> {
        self.relay("get_thinking_level", args)
    }

    /// Relays `set_thinking_level` to the host.
    async fn set_thinking_level(&self, args: Value) -> Result<Value, String> {
        self.relay("set_thinking_level", args)
    }

    /// Relays `compact` to the host.
    async fn compact(&self, args: Value) -> Result<Value, String> {
        self.relay("compact", args)
    }

    /// Relays `get_system_prompt` to the host.
    async fn get_system_prompt(&self, args: Value) -> Result<Value, String> {
        self.relay("get_system_prompt", args)
    }

    /// Relays `new_session` to the host.
    async fn new_session(&self, args: Value) -> Result<Value, String> {
        self.relay("new_session", args)
    }

    /// Relays `fork` to the host.
    async fn fork(&self, args: Value) -> Result<Value, String> {
        self.relay("fork", args)
    }

    /// Relays `navigate_tree` to the host.
    async fn navigate_tree(&self, args: Value) -> Result<Value, String> {
        self.relay("navigate_tree", args)
    }

    /// Relays `switch_session` to the host.
    async fn switch_session(&self, args: Value) -> Result<Value, String> {
        self.relay("switch_session", args)
    }

    /// Relays `reload` to the host.
    async fn reload(&self, args: Value) -> Result<Value, String> {
        self.relay("reload", args)
    }

    /// Relays `get_cli_flag` to the host.
    async fn get_cli_flag(&self, args: Value) -> Result<Value, String> {
        self.relay("get_cli_flag", args)
    }

    /// Relays `ui_dialog` to the host.
    async fn ui_dialog(&self, args: Value) -> Result<Value, String> {
        self.relay("ui_dialog", args)
    }

    /// Relays `get_execution_context` to the host.
    async fn get_execution_context(&self, args: Value) -> Result<Value, String> {
        self.relay("get_execution_context", args)
    }

    /// Relays `child_agent` with its host-owned owner identity.
    async fn child_agent(&self, owner: u64, args: Value) -> Result<Value, String> {
        self.relay("child_agent", json!({"owner": owner, "arguments": args}))
    }

    /// Relays one explicitly allowlisted host capability.
    ///
    /// # Parameters
    /// - `args`: Stable capability name and arguments supplied by the plugin.
    ///
    /// # Returns
    /// The capability result from the host.
    ///
    /// # Errors
    /// Returns a safe denial when the name or capability is unavailable.
    async fn call_host_capability(&self, args: Value) -> Result<Value, String> {
        let capability = args
            .get("capability")
            .and_then(Value::as_str)
            .ok_or_else(|| "host capability name is missing".to_owned())?;
        let arguments = args.get("arguments").cloned().unwrap_or_else(|| json!({}));
        self.relay(capability, arguments)
    }

    /// Relays `get_session_branch` to the host.
    async fn get_session_branch(&self, args: Value) -> Result<Value, String> {
        self.relay("get_session_branch", args)
    }

    /// Relays side-question UI with its host-owned owner identity.
    async fn side_question_ui(
        &self,
        owner: u64,
        thread_id: String,
        args: Value,
    ) -> Result<Value, String> {
        self.relay(
            "side_question_ui",
            json!({"owner": owner, "threadId": thread_id, "arguments": args}),
        )
    }

    /// Reports that side-question streaming is unavailable in this worker.
    async fn side_question(
        &self,
        _owner: u64,
        _args: Value,
        _sink: crate::SideQuestionSink,
    ) -> Result<Value, String> {
        Err("side-question streaming is unavailable in isolated workers".to_owned())
    }
}

/// Loads plugins and serves the worker protocol over standard streams.
///
/// # Parameters
/// - `directories`: Plugin directories selected by the host.
/// - `files`: Explicit plugin files selected by the host.
///
/// # Returns
/// `Ok(())` after orderly shutdown.
///
/// # Errors
/// Returns a load, protection, or protocol error.
pub fn run_plugin_worker(directories: &[PathBuf], files: &[PathBuf]) -> Result<(), String> {
    let worker = PluginWorker::new(directories, files, Arc::new(NullDiagnostics));
    serve_worker_stdio(Arc::new(worker)).map_err(|error| error.to_string())
}

/// Projects a plugin tool result into a bounded protocol value.
///
/// # Parameters
/// - `result`: Plugin-produced agent tool result.
///
/// # Returns
/// JSON content blocks plus details and termination intent.
fn result_to_json(result: &AgentToolResult) -> Value {
    let content = result
        .content
        .iter()
        .filter_map(|content| match content {
            TextContentOrImage::Text(text) => Some(json!({"type": "text", "text": text.text})),
            TextContentOrImage::Image(image) => Some(json!({
                "type": "image",
                "data": image.data,
                "mimeType": image.mime_type,
            })),
        })
        .collect::<Vec<_>>();
    json!({
        "content": content,
        "details": result.details.clone(),
        "terminate": result.terminate,
    })
}
