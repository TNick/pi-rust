//! Fixture sessions and never-driven plugin/host stubs for these tests.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rpi_ai::types::{Api, Tool};
use rpi_ai::Model;
use rpi_plugin_sdk::{
    callbacks::free_string as stub_free_string, FreeStringFn, StbString, StbStringRef, StepHandle,
    StepResult, ToolPartialCb,
};

use crate::{
    ActionBridge, ExtensionRegistry, ExtensionSession, PluginKeepalive, PluginToolHandle,
    RegisteredProvider, RuntimeActionHost,
};

/// Host-owned resource that records when its generation released it.
pub(super) struct Pinned {
    pub(super) dropped: Arc<AtomicBool>,
}

impl Drop for Pinned {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

extern "C" fn unused_execute(
    _tool_call_id: StbStringRef,
    _params: StbString,
    _free_params: Option<FreeStringFn>,
) -> StepHandle {
    std::ptr::null_mut()
}

extern "C" fn unused_poll(
    _handle: StepHandle,
    _partial_cb: Option<ToolPartialCb>,
    _user_data: *mut c_void,
) -> StepResult {
    unreachable!("generation tests never drive a plugin tool")
}

extern "C" fn unused_cancel(_handle: StepHandle) {}

extern "C" fn unused_destroy(_handle: StepHandle) {}

extern "C" fn unused_provider_request(
    _request: StbStringRef,
    _out: *mut StbString,
    _user_data: *mut c_void,
) -> i32 {
    unreachable!("generation tests never invoke a plugin provider callback")
}

/// A session whose one tool is named `tool`, loaded from `plugin_path`.
///
/// # Parameters
/// - `tool`: Tool name to register, used to prove a generation carries its own
///   session.
/// - `plugin_path`: Recorded plugin artifact path (diagnostics only).
///
/// # Returns
/// A session with one registered tool and no action bridge.
pub(super) fn session_with_tool(tool: &str, plugin_path: &str) -> ExtensionSession {
    let mut registry = ExtensionRegistry::new();
    let inserted = registry.register_tool(
        Tool {
            name: tool.to_owned(),
            description: "test tool".to_owned(),
            parameters: rpi_ai::types::Schema::new(serde_json::json!({})),
            constrained_sampling: None,
        },
        PluginToolHandle {
            execute_fn: unused_execute,
            poll_fn: unused_poll,
            cancel_fn: unused_cancel,
            destroy_fn: unused_destroy,
            plugin_free_string: stub_free_string,
            runtime_handle: None,
        },
    );
    assert!(!inserted, "the test tool must be the first registration");
    ExtensionSession::from_parts(
        Arc::new(registry.snapshot()),
        PluginKeepalive::empty(),
        vec![PathBuf::from(plugin_path)],
        None,
    )
}

/// Secret marker the tests assert never reaches a generation identity.
pub(super) const SECRET_ENV: &str = "SCRIBERIG_UPSTREAM_SECRET_ENV";

/// Host endpoint the tests assert never reaches a generation identity.
pub(super) const SECRET_HOST: &str = "secret-invested-host.internal";

/// Private plugin path the tests assert never reaches a generation identity.
pub(super) const SECRET_PATH: &str = "/private/plugins/libpluggy.so";

/// A session that also declares a manifest-backed provider with a credential
/// field and a private endpoint.
///
/// # Returns
/// A session registering provider `pluggy` plus its manifest.
pub(super) fn session_with_provider() -> ExtensionSession {
    let mut registry = ExtensionRegistry::new();
    let declaration = serde_json::json!({
        "providers": [{
            "id": "pluggy",
            "models": [Model::new("m", "Model", Api::Faux, "pluggy", "https://example.test")],
            "apiKeyEnv": SECRET_ENV,
            "requiresAuth": true
        }]
    });
    assert!(registry.apply_declaration_json(&declaration.to_string()));
    registry.register_provider(RegisteredProvider {
        provider_id: "pluggy".to_owned(),
        base_url: format!("https://{SECRET_HOST}/v1"),
        api_style: "faux".to_owned(),
        request_fn: unused_provider_request,
        protected_request_fn: None,
        protected_stream_fn: None,
        stream_fn: None,
        plugin_free_string: stub_free_string,
        user_data: std::ptr::null_mut(),
    });
    ExtensionSession::from_parts(
        Arc::new(registry.snapshot()),
        PluginKeepalive::empty(),
        vec![PathBuf::from(SECRET_PATH)],
        None,
    )
}

/// A session whose plugins can call back into the host through `bridge`.
///
/// # Parameters
/// - `tool`: Tool name to register.
/// - `bridge`: Action bridge to attach to the session.
///
/// # Returns
/// A session sharing that bridge.
pub(super) fn session_with_bridge(tool: &str, bridge: &Arc<ActionBridge>) -> ExtensionSession {
    let session = session_with_tool(tool, "/plugins/first.so");
    ExtensionSession::from_parts(
        session
            .snapshot_arc()
            .expect("the test session registers a tool"),
        session.keepalive(),
        vec![PathBuf::from("/plugins/first.so")],
        Some(Arc::clone(bridge)),
    )
}

/// A host stub that is never driven; only bridge staleness is observed.
#[derive(Default)]
pub(super) struct UnusedHost;

#[async_trait::async_trait]
impl RuntimeActionHost for UnusedHost {
    async fn send_message(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn send_user_message(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn append_entry(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn set_session_name(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn get_active_tools(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn set_active_tools(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn set_model(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn get_thinking_level(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn set_thinking_level(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn compact(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn get_system_prompt(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn new_session(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn fork(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn navigate_tree(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn switch_session(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
    async fn reload(&self, _: serde_json::Value) -> Result<serde_json::Value, String> {
        Err("unused in generation tests".into())
    }
}
