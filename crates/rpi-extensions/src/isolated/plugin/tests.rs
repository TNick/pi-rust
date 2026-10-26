//! Verifies isolated plugin provider calls retain sanitized applied settings.

use std::sync::{Arc, Mutex};

use rpi_ai::{
    Api, AppliedRequestSettings, AppliedRequestSettingsCallback, AssistantMessage,
    AssistantMessageEvent, Context, Model, Provider, RequestSecret, SimpleStreamOptions,
    StopReason,
};
use tokio_util::sync::CancellationToken;

use super::{LoadedPlugins, PluginWorker};
use crate::NullDiagnostics;
use crate::actions::{ActionBridge, RuntimeActionHost};
use crate::isolated::protocol::{
    WorkerProviderOptions, WorkerProviderRequest, WorkerProviderResponse,
};
use crate::isolated::worker::WorkerHandler;
use crate::loader::PluginKeepalive;

/// Supplies fixed provider reports to the worker call boundary.
struct ReportingProvider {
    /// Exact model advertised to the worker host.
    model: Model,
    /// Sanitized report delivered by either callback variant.
    report: AppliedRequestSettings,
    /// Records which request methods the worker invoked.
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait::async_trait]
impl Provider for ReportingProvider {
    /// Returns the fixture provider identity.
    fn id(&self) -> &str {
        &self.model.provider
    }

    /// Returns the one fixture model.
    fn models(&self) -> &[Model] {
        std::slice::from_ref(&self.model)
    }

    /// Returns one completed fixture response without a settings callback.
    async fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
    ) -> rpi_ai::AssistantMessageEventStream {
        completed_stream(model)
    }

    /// Reports the fixture settings for an ordinary request.
    async fn stream_simple_with_applied_settings(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
        callback: AppliedRequestSettingsCallback,
    ) -> rpi_ai::AssistantMessageEventStream {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push("ordinary");
        callback(self.report.clone());
        completed_stream(model)
    }

    /// Reports the fixture settings for a protected request.
    async fn stream_simple_with_request_secret_and_applied_settings(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
        _secret: &RequestSecret,
        _bearer: bool,
        callback: AppliedRequestSettingsCallback,
    ) -> rpi_ai::AssistantMessageEventStream {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push("protected");
        callback(self.report.clone());
        completed_stream(model)
    }
}

/// Returns a terminal fixture stream for the selected model.
///
/// # Parameters
/// - `model`: Provider model for the terminal message identity.
///
/// # Returns
/// An event stream that finishes with one completed assistant message.
fn completed_stream(model: &Model) -> rpi_ai::AssistantMessageEventStream {
    let (mut producer, stream) = rpi_ai::event_stream::create_assistant_message_event_stream();
    producer.push(AssistantMessageEvent::Done {
        reason: rpi_ai::types::DoneReason::Stop,
        message: AssistantMessage::terminal(
            model.api.clone(),
            model.provider.clone(),
            model.id.clone(),
            StopReason::Stop,
            "fixture response",
            1,
        ),
    });
    stream
}

/// Builds a worker that owns the reporting provider fixture.
///
/// # Returns
/// A loaded worker with one provider and a live Tokio runtime.
fn worker() -> (PluginWorker, Arc<Mutex<Vec<&'static str>>>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("fixture runtime");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = Model::new(
        "model",
        "Model",
        Api::Faux,
        "provider",
        "https://provider.example.test/v1",
    );
    let provider = Arc::new(ReportingProvider {
        report: AppliedRequestSettings {
            schema_version: "rpi-applied-request-settings-v5".to_owned(),
            provider_id: "provider".to_owned(),
            model_id: "model".to_owned(),
            api_id: Api::Faux.as_str().to_owned(),
            applied_values: Default::default(),
            request_body_sha256: None,
            request_header_sha256: None,
            request_effect_sha256: Some("safe-effect-digest".to_owned()),
            provider_options: Default::default(),
            omitted_categories: Vec::new(),
            complete: true,
        },
        model,
        calls: Arc::clone(&calls),
    });
    let bridge = ActionBridge::new(runtime.handle().clone(), Arc::new(EmptyHost));
    let worker = PluginWorker {
        directories: Vec::new(),
        files: Vec::new(),
        diagnostics: Arc::new(NullDiagnostics),
        loaded: Mutex::new(Some(LoadedPlugins {
            keepalive: PluginKeepalive::empty(),
            tools: Vec::new(),
            commands: Vec::new(),
            providers: vec![provider],
            loaded_paths: Vec::new(),
            runtime,
            _bridge: bridge,
        })),
    };
    (worker, calls)
}

/// Implements a no-op host for requests that do not use runtime actions.
struct EmptyHost;

macro_rules! unsupported_actions {
    ($($name:ident),+ $(,)?) => {
        #[async_trait::async_trait]
        impl RuntimeActionHost for EmptyHost {
            $(
                async fn $name(
                    &self,
                    _args: serde_json::Value,
                ) -> Result<serde_json::Value, String> {
                    Err("fixture host action is unused".to_owned())
                }
            )+
        }
    };
}

unsupported_actions!(
    send_message,
    send_user_message,
    append_entry,
    set_session_name,
    get_active_tools,
    set_active_tools,
    set_model,
    get_thinking_level,
    set_thinking_level,
    compact,
    get_system_prompt,
    new_session,
    fork,
    navigate_tree,
    switch_session,
    reload,
);

/// Proves both provider callback variants survive the isolated worker call.
#[test]
fn provider_call_returns_applied_settings_for_ordinary_and_protected_requests() {
    let (worker, calls) = worker();
    for secret in [None, Some("credential".to_owned())] {
        let request = WorkerProviderRequest {
            provider: "provider".to_owned(),
            model: "model".to_owned(),
            context: Context::default(),
            options: WorkerProviderOptions::default(),
            secret: secret.clone(),
            bearer: secret.is_some(),
        };
        let response: WorkerProviderResponse = worker
            .call_provider(request, CancellationToken::new())
            .expect("provider response");
        assert_eq!(
            response.outcome,
            super::super::protocol::WorkerProviderOutcome::Stop
        );
        assert_eq!(
            response
                .applied_settings
                .and_then(|settings| settings.request_effect_sha256),
            Some("safe-effect-digest".to_owned())
        );
    }
    assert_eq!(
        *calls.lock().expect("provider call log"),
        ["ordinary", "protected"]
    );
}
