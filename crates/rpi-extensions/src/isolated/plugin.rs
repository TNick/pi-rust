//! Loads and executes native plugins only inside an isolated worker process.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rpi_agent::agent_tool::AgentTool;
use rpi_agent::types::{AgentToolResult, TextContentOrImage};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::actions::{ActionBridge, RuntimeActionHost};
use crate::loader::{load_session_mixed, PluginKeepalive};
use crate::registry::ExtensionTool;
use crate::tool::PluginToolAdapter;
use crate::{NullDiagnostics, PluginDiagnostics};

use super::protocol::WorkerToolSpec;
use super::worker::{serve_worker_stdio, WorkerHandler, WorkerHostClient};

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
        let bridge = ActionBridge::new(runtime.handle().clone(), action_host);
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
        if tools.is_empty() {
            return Err("no plugin tool registered in the worker".to_owned());
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
