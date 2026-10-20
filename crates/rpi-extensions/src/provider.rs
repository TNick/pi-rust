//! B5c — `PluggableProvider`: the host-side `Provider` impl wrapping a plugin's
//! `ProviderRequestFn`.
//!
//! A plugin's [`ProviderRequestFn`](rpi_plugin_sdk::ProviderRequestFn) is a
//! **synchronous one-shot**: given a borrowed JSON request envelope it writes a
//! **plugin-owned** `out` [`StbString`] (a full assistant-message JSON). It
//! cannot drive a chunked `stream_simple` without blocking, so the v1
//! `PluggableProvider` emits the full response as a **single terminal `Done`
//! chunk** — a documented divergence from pi's async `streamSimple`
//! (`custom-provider.md`). The ffi call runs on `spawn_blocking` (via a captured
//! [`Handle`](tokio::runtime::Handle) — same foreign-thread fix as the
//! `runtime_action` bridge), so the async `stream_simple` never blocks a runtime
//! worker.
//!
//! ## Request/response envelope
//!
//! - **Request** (borrowed `StbStringRef` handed to `request_fn`): JSON
//!   `{ model, context, options }`, where `model`/`context` are the serde
//!   serializations of [`Model`](rpi_ai::Model)/[`Context`](rpi_ai::Context)
//!   and `options` is a hand-built JSON of the [`SimpleStreamOptions`] fields
//!   (that type is not `Serialize`; we mirror the field set
//!   [`ExtensionProviderHooks`](crate::ExtensionProviderHooks) dispatches). A
//!   non-empty `providerOptions` field is sent only to a provider whose
//!   declaration opts into version 1; other providers fail closed.
//! - **Response** (the plugin-owned `out` `StbString`): an
//!   [`AssistantMessage`] JSON. The host reclaims `out` via the plugin's
//!   `plugin_free_string` (traveled with the registration), parses the JSON, and
//!   pushes it as one `Done` (or `Error` on nonzero rc / parse failure).
//!
//! ## Lifetime + staleness
//!
//! `PluggableProvider` mirrors [`ExtensionProviderHooks`]: it holds an
//! `Arc<RegistrySnapshot>` (the staleness flag is shared — a `/reload`
//! invalidate marks it stale) + an `Arc<PluginKeepalive>` (keeps the cdylib
//! mapped so `request_fn` stays callable) + the [`RegisteredProvider`] record +
//! a `Handle` captured at build time (the host is on the runtime when it
//! constructs providers from the session).

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rpi_ai::event_stream::create_assistant_message_event_stream;
use rpi_ai::types::{AssistantMessage, Content, DoneReason, ErrorReason, ToolCall, ToolCallType};
use rpi_ai::{
    AppliedRequestSettings, AppliedRequestSettingsCallback, AssistantMessageEvent,
    AssistantMessageEventStream, AssistantMessageEventStreamProducer, Context, Model, Provider,
    RequestSecret, SecretMaterial, SimpleStreamOptions,
};
use rpi_plugin_sdk::callbacks::OwnedString;
use rpi_plugin_sdk::{StbString, StbStringRef};
use tokio::runtime::Handle;

use crate::loader::ExtensionSession;
use crate::registry::{RegisteredProvider, RegistrySnapshot};
use crate::PluginKeepalive;

#[cfg(test)]
mod lifetime_tests;
#[cfg(test)]
mod stream_tests;

/// Reserve child capacity before spawning the producer; move this ownership
/// into the foreign driver so runtime shutdown cannot unload executing code.
struct ProviderRequestLifetime {
    _keepalive: Arc<PluginKeepalive>,
    _background: Option<rpi_agent::tool_execution_scope::BackgroundToolGuard>,
}

/// Owns a legacy options credential while routing it over the protected path.
///
/// Attributes:
/// - `0`: Credential string retained for the synchronous plugin request.
struct OptionsSecret(String);

impl SecretMaterial for OptionsSecret {
    /// Lend the credential only for the synchronous consumer callback.
    ///
    /// # Parameters
    /// - `consumer`: Request code that consumes the credential.
    fn with_secret(&self, consumer: &mut dyn FnMut(&str)) {
        consumer(&self.0);
    }
}
impl ProviderRequestLifetime {
    fn new(keepalive: Arc<PluginKeepalive>) -> Self {
        Self {
            _keepalive: keepalive,
            _background: rpi_agent::tool_execution_scope::track_background_tool(),
        }
    }
}

/// A `Provider` backed by a plugin's sync `ProviderRequestFn`.
///
/// See the module docs for the v1 one-shot streaming model + the request/response
/// envelope. Built from a loaded [`ExtensionSession`]; one `PluggableProvider`
/// per [`RegisteredProvider`] in the session's snapshot.
pub struct PluggableProvider {
    /// The registration record (id/base_url/api_style + the fn pointers). Cloned
    /// from the snapshot; the fn pointers are valid for the keepalive's lifetime.
    record: RegisteredProvider,
    /// Shared staleness flag (a `/reload` invalidate flows through here).
    snapshot: Arc<RegistrySnapshot>,
    /// Keeps the cdylib mapped so `request_fn` stays callable.
    _keepalive: Arc<PluginKeepalive>,
    /// Captured at build time — `spawn_blocking` works from the async context.
    runtime: Handle,
    models: Vec<Model>,
    provider_options_v1: bool,
}

impl PluggableProvider {
    fn new(
        record: RegisteredProvider,
        snapshot: Arc<RegistrySnapshot>,
        keepalive: Arc<PluginKeepalive>,
        runtime: Handle,
    ) -> Arc<Self> {
        let manifest = snapshot.provider_manifest(&record.provider_id);
        let models = manifest
            .map(|manifest| manifest.models.clone())
            .unwrap_or_default();
        let provider_options_v1 =
            manifest.is_some_and(|manifest| manifest.provider_options_version == Some(1));
        Arc::new(Self {
            record,
            snapshot,
            _keepalive: keepalive,
            runtime,
            models,
            provider_options_v1,
        })
    }

    /// Build one [`PluggableProvider`] per registered provider in the session's
    /// snapshot, as `Arc<dyn Provider>`. Returns an empty vec when no plugin
    /// registered a provider (so a session without provider plugins injects
    /// nothing). The `Handle` MUST be captured from a thread running the target
    /// runtime (pi-cli builds providers on the async main thread, same as the
    /// `ActionBridge`).
    pub fn from_session(session: &ExtensionSession, runtime: Handle) -> Vec<Arc<dyn Provider>> {
        let Some(snapshot) = session.snapshot_arc() else {
            return Vec::new();
        };
        let keepalive = session.keepalive();
        snapshot
            .providers()
            .iter()
            .cloned()
            .map(|record| {
                Self::new(
                    record,
                    Arc::clone(&snapshot),
                    Arc::clone(&keepalive),
                    runtime.clone(),
                ) as Arc<dyn Provider>
            })
            .collect()
    }

    /// Stream a provider request with an opaque host-protected credential.
    ///
    /// # Parameters
    /// - `model`: Model to execute.
    /// - `ctx`: Prompt context.
    /// - `opts`: Non-secret request options.
    /// - `secret`: Opaque credential handle.
    ///
    /// # Returns
    /// The provider response stream, or an error stream if this extension uses
    /// the one-shot ABI, which has no protected credential channel.
    async fn stream_with_secret(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        secret: Option<(RequestSecret, bool)>,
        applied_settings_callback: Option<AppliedRequestSettingsCallback>,
    ) -> AssistantMessageEventStream {
        let (mut prod, stream) = create_assistant_message_event_stream();
        let snapshot = Arc::clone(&self.snapshot);
        let record = self.record.clone();
        let runtime = self.runtime.clone();
        let lifetime = ProviderRequestLifetime::new(Arc::clone(&self._keepalive));
        let model = model.clone();
        let ctx = Arc::new(ctx.clone());
        let mut opts = opts.clone();
        opts.api_key = None;
        let provider_options_v1 = self.provider_options_v1;

        tokio::spawn(async move {
            let message = if !provider_options_v1 && !model.provider_options.is_empty() {
                AssistantMessage::terminal(
                    model.api.clone(),
                    record.provider_id.clone(),
                    model.id.clone(),
                    rpi_ai::types::StopReason::Error,
                    "extension provider does not declare providerOptions version 1",
                    0,
                )
            } else if let Some(stream_fn) = record.protected_stream_fn {
                drive_streaming_plugin_provider(
                    &snapshot,
                    &record,
                    StreamProviderCallback::Protected(stream_fn),
                    runtime,
                    lifetime,
                    &model,
                    &ctx,
                    &opts,
                    secret,
                    &mut prod,
                )
                .await
            } else if let Some(stream_fn) = record
                .stream_fn
                .filter(|_| secret.is_none() && !request_has_headers(&model, &opts))
            {
                drive_streaming_plugin_provider(
                    &snapshot,
                    &record,
                    StreamProviderCallback::Legacy(stream_fn),
                    runtime,
                    lifetime,
                    &model,
                    &ctx,
                    &opts,
                    secret,
                    &mut prod,
                )
                .await
            } else {
                match (secret, record.protected_request_fn) {
                    (Some((secret, bearer)), Some(request_fn)) => {
                        drive_protected_plugin_provider(
                            &snapshot,
                            &record,
                            request_fn,
                            runtime,
                            lifetime,
                            &model,
                            &ctx,
                            &opts,
                            Some(secret),
                            bearer,
                        )
                        .await
                    }
                    (None, Some(request_fn)) => {
                        drive_protected_plugin_provider(
                            &snapshot, &record, request_fn, runtime, lifetime, &model, &ctx, &opts,
                            None, false,
                        )
                        .await
                    }
                    (Some(_), None) => credential_unsupported_message(&model, &record.provider_id),
                    (None, None) if request_has_headers(&model, &opts) => {
                        credential_unsupported_message(&model, &record.provider_id)
                    }
                    (None, None) => {
                        drive_plugin_provider(
                            &snapshot, &record, runtime, lifetime, &model, &ctx, &opts,
                        )
                        .await
                    }
                }
            };
            if let Some(callback) = applied_settings_callback {
                callback(incomplete_applied_settings_report(&model));
            }
            push_terminal(&mut prod, message);
        });
        stream
    }
}

#[async_trait::async_trait]
impl Provider for PluggableProvider {
    fn id(&self) -> &str {
        &self.record.provider_id
    }

    fn models(&self) -> &[Model] {
        &self.models
    }

    async fn stream_simple(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let secret = opts
            .api_key
            .as_ref()
            .map(|key| RequestSecret::new(Arc::new(OptionsSecret(key.clone()))));
        self.stream_with_secret(model, ctx, opts, secret.map(|secret| (secret, false)), None)
            .await
    }

    async fn stream_simple_with_request_secret(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
    ) -> AssistantMessageEventStream {
        self.stream_with_secret(model, ctx, opts, Some((secret.clone(), bearer)), None)
            .await
    }

    async fn stream_simple_with_applied_settings(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        let secret = opts
            .api_key
            .as_ref()
            .map(|key| RequestSecret::new(Arc::new(OptionsSecret(key.clone()))));
        self.stream_with_secret(
            model,
            ctx,
            opts,
            secret.map(|secret| (secret, false)),
            Some(callback),
        )
        .await
    }

    async fn stream_simple_with_request_secret_and_applied_settings(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        self.stream_with_secret(
            model,
            ctx,
            opts,
            Some((secret.clone(), bearer)),
            Some(callback),
        )
        .await
    }
}

/// Builds an explicitly incomplete report for an opaque extension request.
///
/// The host knows the selected public model identity but cannot observe plugin
/// request transformations or determine which configuration values were used.
///
/// # Parameters
/// - `model`: Selected provider model descriptor.
///
/// # Returns
/// An incomplete report with no asserted applied settings.
fn incomplete_applied_settings_report(model: &Model) -> AppliedRequestSettings {
    AppliedRequestSettings {
        schema_version: "rpi-applied-request-settings-v1".to_owned(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        api_id: model.api.as_str().to_owned(),
        applied_values: Default::default(),
        omitted_categories: vec![
            "extension_managed_request_transforms".to_owned(),
            "provider_applied_values_unavailable".to_owned(),
            "provider_options_values".to_owned(),
            "endpoint".to_owned(),
            "headers_and_credentials".to_owned(),
            "prompt_and_transcript".to_owned(),
            "tools_and_schemas".to_owned(),
            "unrecognized_request_fields".to_owned(),
        ],
        complete: false,
    }
}

/// Run the sync `ProviderRequestFn` on `spawn_blocking`, read its plugin-owned
/// `out` JSON, reclaim it via `plugin_free_string`, parse to an
/// [`AssistantMessage`]. Returns an error terminal message on any failure
/// (staleness, nonzero rc, parse error, runtime shutdown).
async fn drive_plugin_provider(
    snapshot: &Arc<RegistrySnapshot>,
    record: &RegisteredProvider,
    runtime: Handle,
    lifetime: ProviderRequestLifetime,
    model: &Model,
    ctx: &Context,
    opts: &SimpleStreamOptions,
) -> AssistantMessage {
    // Staleness: a `/reload`d session must not drive its old providers.
    if !snapshot.is_active() {
        return AssistantMessage::terminal(
            model.api.clone(),
            record.provider_id.clone(),
            model.id.clone(),
            rpi_ai::types::StopReason::Error,
            "extensions provider registry is stale (session swapped/reloaded)",
            0,
        );
    }

    let request_json = match request_envelope_json(model, ctx, opts) {
        Ok(request_json) => request_json,
        Err(error) => {
            return AssistantMessage::terminal(
                model.api.clone(),
                record.provider_id.clone(),
                model.id.clone(),
                rpi_ai::types::StopReason::Error,
                error,
                0,
            );
        }
    };

    // The fn pointers + user_data are `Copy`-able (fn ptrs) or plain raw (ud);
    // clone what spawn_blocking needs. `record` is `Clone` (owned strings).
    let record = record.clone();
    let provider_id = record.provider_id.clone();
    let api = model.api.clone();
    let model_id = model.id.clone();

    // `spawn_blocking` the sync ffi call — `request_fn` is blocking by contract.
    // The join handle is awaited so a panic in the plugin (caught by
    // `catch_unwind` below) surfaces as an error, not a silent hang.
    let request_model_id = model_id.clone();
    let join = runtime.spawn_blocking(move || {
        let _lifetime = lifetime;
        run_provider_request(&record, &request_json, &request_model_id)
    });
    let outcome = match join.await {
        Ok(inner) => inner,
        Err(join_err) => {
            // The blocking task panicked before returning (catch_unwind inside
            // caught a plugin panic and aborted, OR tokio cancelled). Either way
            // no response.
            return AssistantMessage::terminal(
                api,
                provider_id,
                model_id,
                rpi_ai::types::StopReason::Error,
                format!("plugin provider task failed: {join_err}"),
                0,
            );
        }
    };

    match outcome {
        ProviderOutcome::Ok(message) => message,
        ProviderOutcome::Err(msg) => AssistantMessage::terminal(
            api,
            provider_id,
            model_id,
            rpi_ai::types::StopReason::Error,
            msg,
            0,
        ),
    }
}

/// Drive a credentialed one-shot provider through its protected ABI callback.
///
/// # Parameters
/// - `snapshot`: Registry snapshot used to reject stale provider instances.
/// - `record`: Registered extension provider and ownership callbacks.
/// - `request_fn`: Protected one-shot ABI callback.
/// - `runtime`: Runtime used to execute the blocking plugin callback.
/// - `lifetime`: Plugin lifetime guard held until the callback returns.
/// - `model`: Selected model.
/// - `ctx`: Prompt context.
/// - `opts`: Non-secret request options.
/// - `secret`: Optional protected request credential.
/// - `bearer`: Whether authorization uses bearer semantics.
///
/// # Returns
/// The plugin response or an error terminal message.
async fn drive_protected_plugin_provider(
    snapshot: &Arc<RegistrySnapshot>,
    record: &RegisteredProvider,
    request_fn: rpi_plugin_sdk::ProtectedProviderRequestFn,
    runtime: Handle,
    lifetime: ProviderRequestLifetime,
    model: &Model,
    ctx: &Context,
    opts: &SimpleStreamOptions,
    secret: Option<RequestSecret>,
    bearer: bool,
) -> AssistantMessage {
    if !snapshot.is_active() {
        return AssistantMessage::terminal(
            model.api.clone(),
            record.provider_id.clone(),
            model.id.clone(),
            rpi_ai::types::StopReason::Error,
            "extensions provider registry is stale (session swapped/reloaded)",
            0,
        );
    }

    let request_json = match request_envelope_json(model, ctx, opts) {
        Ok(request_json) => request_json,
        Err(error) => {
            return AssistantMessage::terminal(
                model.api.clone(),
                record.provider_id.clone(),
                model.id.clone(),
                rpi_ai::types::StopReason::Error,
                error,
                0,
            );
        }
    };
    let record = record.clone();
    let provider_id = record.provider_id.clone();
    let api = model.api.clone();
    let model_id = model.id.clone();
    let request_model_id = model_id.clone();
    let snapshot = Arc::clone(snapshot);
    let owner = Arc::new(ProviderControlOwner {
        state: Arc::new(ProviderControlState::default()),
        secret: secret.map(|secret| (secret, bearer)),
        headers: protected_headers(model, opts),
    });
    let join = runtime.spawn_blocking(move || {
        let _lifetime = lifetime;
        run_protected_provider_request(
            &snapshot,
            &record,
            request_fn,
            &request_json,
            &request_model_id,
            &owner,
        )
    });

    match join.await {
        Ok(ProviderOutcome::Ok(message)) => message,
        Ok(ProviderOutcome::Err(message)) => AssistantMessage::terminal(
            api,
            provider_id,
            model_id,
            rpi_ai::types::StopReason::Error,
            message,
            0,
        ),
        Err(join_error) => AssistantMessage::terminal(
            api,
            provider_id,
            model_id,
            rpi_ai::types::StopReason::Error,
            format!("plugin provider task failed: {join_error}"),
            0,
        ),
    }
}

/// Invoke a protected one-shot provider on the blocking pool and parse its reply.
///
/// # Parameters
/// - `snapshot`: Registry snapshot used for a final staleness check.
/// - `record`: Registered provider and plugin-owned output deallocator.
/// - `request_fn`: Protected plugin callback.
/// - `request_json`: Non-secret serialized request envelope.
/// - `model_id`: Selected model identity for response defaults.
/// - `owner`: Credential/control owner held for the callback duration.
///
/// # Returns
/// Parsed response or an error description.
fn run_protected_provider_request(
    snapshot: &Arc<RegistrySnapshot>,
    record: &RegisteredProvider,
    request_fn: rpi_plugin_sdk::ProtectedProviderRequestFn,
    request_json: &str,
    model_id: &str,
    owner: &ProviderControlOwner,
) -> ProviderOutcome {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !snapshot.is_active() {
            return ProviderOutcome::Err("extensions provider registry is stale".into());
        }
        let req_ref = StbStringRef::from_str(request_json);
        let mut out = StbString::empty();
        let protected_control = protected_control(owner);
        let rc = request_fn(
            req_ref,
            &protected_control,
            &mut out as *mut StbString,
            record.user_data,
        );
        let out = unsafe { OwnedString::from_raw(out, Some(record.plugin_free_string)) };
        if rc != 0 {
            return ProviderOutcome::Err(format!(
                "plugin provider {} returned error code {rc}",
                record.provider_id
            ));
        }
        let response_text = out.to_string_lossy();
        match serde_json::from_str::<AssistantMessage>(&response_text) {
            Ok(mut message) => {
                if message.provider.is_empty() {
                    message.provider = record.provider_id.clone();
                }
                if message.model.is_empty() {
                    message.model = model_id.to_string();
                }
                ProviderOutcome::Ok(message)
            }
            Err(error) => ProviderOutcome::Err(format!(
                "plugin provider {} returned unparseable response: {error}",
                record.provider_id
            )),
        }
    }))
    .unwrap_or_else(|_| {
        ProviderOutcome::Err(format!(
            "plugin provider {} panicked during protected request",
            record.provider_id
        ))
    })
}

/// The result of one sync `request_fn` call: either a parsed assistant message
/// or an error string. Computed inside `spawn_blocking`.
enum ProviderOutcome {
    Ok(AssistantMessage),
    Err(String),
}

/// The inner sync driver (runs on the blocking pool). Calls `request_fn`,
/// recovers the plugin-owned `out` `StbString`, reclaims it via
/// `plugin_free_string`, parses to an [`AssistantMessage`]. Wrapped in
/// `catch_unwind` so a plugin panic cannot unwind across FFI (⇒ error outcome).
fn run_provider_request(
    record: &RegisteredProvider,
    request_json: &str,
    model_id: &str,
) -> ProviderOutcome {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_provider_request_inner(record, request_json, model_id)
    }));
    match outcome {
        Ok(inner) => inner,
        Err(_) => {
            tracing::error!(
                "plugin provider {} panicked — refusing to unwind across FFI",
                record.provider_id
            );
            ProviderOutcome::Err(format!(
                "plugin provider {} panicked during request",
                record.provider_id
            ))
        }
    }
}

fn run_provider_request_inner(
    record: &RegisteredProvider,
    request_json: &str,
    model_id: &str,
) -> ProviderOutcome {
    // Borrowed request envelope: the plugin must NOT free it (StbStringRef is a
    // borrow — the SDK contract).
    let req_ref = StbStringRef::from_str(request_json);

    // Plugin-owned `out` — `request_fn` writes a StbString into `*out` (plugin
    // allocation, reclaimed via `plugin_free_string`). Start empty.
    let mut out = StbString::empty();
    let rc = (record.request_fn)(req_ref, &mut out as *mut StbString, record.user_data);
    // SAFETY: the callback transfers its live output to its registered producer.
    let out = unsafe {
        rpi_plugin_sdk::callbacks::OwnedString::from_raw(out, Some(record.plugin_free_string))
    };
    if rc != 0 {
        // Nonzero rc: the plugin signaled failure. Reclaim whatever it wrote
        // (may be empty) before returning.
        return ProviderOutcome::Err(format!(
            "plugin provider {} returned error code {rc}",
            record.provider_id
        ));
    }

    // Read the plugin-owned JSON (borrow `out` for the parse) then reclaim it.
    let response_text = out.to_string_lossy();

    // Parse the response as an AssistantMessage. A plugin that returns a partial
    // or malformed message surfaces a parse error (not a crash) — the host
    // emits it as an `Error` terminal so the run terminates cleanly.
    match serde_json::from_str::<AssistantMessage>(&response_text) {
        Ok(message) => {
            // Stamp the provider/model identity the host expects if the plugin
            // omitted them (a plugin may return only `content`/`stopReason`).
            // We do NOT overwrite fields the plugin set — only fill defaults on
            // a message whose provider/model don't match this provider.
            let mut message = message;
            if message.provider.is_empty() {
                message.provider = record.provider_id.clone();
            }
            if message.model.is_empty() {
                message.model = model_id.to_string();
            }
            ProviderOutcome::Ok(message)
        }
        Err(err) => ProviderOutcome::Err(format!(
            "plugin provider {} returned unparseable response: {err}",
            record.provider_id
        )),
    }
}

/// Push the terminal event for `message`. A `StopReason::Error`/`Aborted`
/// message → `Error` (with the matching `ErrorReason`); otherwise → `Done`
/// (mapping the stop reason onto [`DoneReason`]). This is the single chunk the
/// one-shot path emits, and the terminal event for the streaming path.
fn push_terminal(prod: &mut AssistantMessageEventStreamProducer, message: AssistantMessage) {
    use rpi_ai::types::StopReason;
    match message.stop_reason {
        // The two failure reasons map to distinct `ErrorReason`s: `Aborted` is
        // the caller going away (a cancellation), not a provider fault. The agent
        // loop and the TUI distinguish them, so collapsing both onto `Error`
        // would report a user-initiated cancel as a provider failure.
        StopReason::Error | StopReason::Aborted => {
            let reason = if message.stop_reason == StopReason::Aborted {
                ErrorReason::Aborted
            } else {
                ErrorReason::Error
            };
            prod.push(AssistantMessageEvent::Error {
                reason,
                error: message,
            });
        }
        other => {
            let reason = match other {
                StopReason::Stop => DoneReason::Stop,
                StopReason::Length => DoneReason::Length,
                StopReason::ToolUse => DoneReason::ToolUse,
                StopReason::Deferred => DoneReason::Deferred,
                // Pending/unknown settle to Stop (the plugin should have set a
                // terminal reason; if not, the response is still complete).
                _ => DoneReason::Stop,
            };
            prod.push(AssistantMessageEvent::Done { reason, message });
        }
    }
}

/// Build the `options` JSON for the request envelope. `SimpleStreamOptions` is
/// not `Serialize`, so we hand-build the field set (mirrors
/// `ExtensionProviderHooks::before_request`). Sensitive fields (`api_key`) are
/// excluded — provider credentials are exposed only through the protected
/// streaming control callback, never through serialized JSON.
fn options_json(opts: &SimpleStreamOptions) -> serde_json::Value {
    serde_json::json!({
        "timeoutMs": opts.timeout.map(|d| d.as_millis() as u64),
        "maxRetries": opts.max_retries,
        "maxRetryDelayMs": opts.max_retry_delay.map(|d| d.as_millis() as u64),
        "metadata": opts.metadata,
        "cacheRetention": format!("{:?}", opts.cache_retention),
        "sessionId": opts.session_id,
        "reasoning": opts.reasoning.map(|r| format!("{r:?}")),
        "maxTokens": opts.max_tokens,
        "temperature": opts.temperature,
        "thinkingBudgets": opts.thinking_budgets,
    })
}

/// Return whether request headers require a protected provider ABI.
///
/// # Parameters
/// - `model`: Selected model with optional model-level headers.
/// - `opts`: Per-request options with optional request headers.
///
/// # Returns
/// `true` when either header map contains entries.
fn request_has_headers(model: &Model, opts: &SimpleStreamOptions) -> bool {
    model
        .headers
        .as_ref()
        .is_some_and(|headers| !headers.is_empty())
        || opts
            .headers
            .as_ref()
            .is_some_and(|headers| !headers.is_empty())
}

/// Construct a terminal error when the extension ABI cannot consume a secret.
fn credential_unsupported_message(model: &Model, provider_id: &str) -> AssistantMessage {
    AssistantMessage::terminal(
        model.api.clone(),
        provider_id,
        model.id.clone(),
        rpi_ai::types::StopReason::Error,
        "extension provider ABI does not support protected request inputs",
        0,
    )
}

// ---------------------------------------------------------------------------
// Streaming providers (B5c-s) — the per-chunk callback bridge
// ---------------------------------------------------------------------------

/// Run a plugin's [`ProviderStreamRequestFn`](rpi_plugin_sdk::ProviderStreamRequestFn)
/// on `spawn_blocking` while forwarding every emitted chunk to `prod` **as it
/// arrives**, then return the plugin's terminal message.
///
/// The forwarding is the whole point: the request fn is synchronous, so the
/// callback is the only moment the host learns about a delta. It runs on the
/// blocking-pool thread, inside the plugin's call, and queues chunks for the
/// async driver to forward — which is what makes the consumer's `next()`
/// resolve while the model is still producing.
///
/// Ordering guarantee: chunks are pushed from the single blocking thread, in the
/// order the plugin emits them. On completion the caller waits for the plugin's
/// final message; on cancellation it closes the chunk channel immediately. No
/// chunk can therefore land after the terminal event.
///
/// Cancellation: the caller's signal is selected against incoming chunks and
/// the final join. When it fires the driver (a) latches the control state so the
/// plugin can see it, (b) drops the chunk channel so the plugin's next `emit`
/// reports a rejected chunk, and (c) returns immediately with an `Aborted`
/// terminal rather than waiting out a generation nobody will read. The blocking
/// call itself cannot be interrupted mid-FFI; a plugin that honors the latch
/// unwinds promptly, and one that does not still terminates the run.
#[allow(clippy::too_many_arguments)]
async fn drive_streaming_plugin_provider(
    snapshot: &Arc<RegistrySnapshot>,
    record: &RegisteredProvider,
    stream_fn: StreamProviderCallback,
    runtime: Handle,
    lifetime: ProviderRequestLifetime,
    model: &Model,
    ctx: &Context,
    opts: &SimpleStreamOptions,
    secret: Option<(RequestSecret, bool)>,
    prod: &mut AssistantMessageEventStreamProducer,
) -> AssistantMessage {
    // Staleness: a `/reload`d session must not drive its old providers.
    if !snapshot.is_active() {
        return AssistantMessage::terminal(
            model.api.clone(),
            record.provider_id.clone(),
            model.id.clone(),
            rpi_ai::types::StopReason::Error,
            "extensions provider registry is stale (session swapped/reloaded)",
            0,
        );
    }

    // `options.signal` is the same cancellation contract the built-in providers
    // honor: the agent loop cancels this token when the user aborts the run.
    let cancel = opts.signal.clone();
    let request_json = match request_envelope_json(model, ctx, opts) {
        Ok(request_json) => request_json,
        Err(error) => {
            return AssistantMessage::terminal(
                model.api.clone(),
                record.provider_id.clone(),
                model.id.clone(),
                rpi_ai::types::StopReason::Error,
                error,
                0,
            );
        }
    };

    // The callback bridge. It is moved (not shared) into the blocking task, and
    // it owns the chunk channel's **only** sender: when that task drops it, the
    // consumer's `recv()` sees the channel close. That lifetime coupling is the
    // reason this is a plain field and not a second `Arc` handle.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamChunk>();
    let control = Arc::new(ProviderControlState::default());
    let bridge = StreamEmitBridge {
        tx,
        plugin_free_string: record.plugin_free_string,
        control: Some(Arc::clone(&control)),
    };
    let owner = Arc::new(ProviderControlOwner {
        state: Arc::clone(&control),
        secret,
        headers: protected_headers(model, opts),
    });

    let record_owned = record.clone();
    let snapshot_owned = Arc::clone(snapshot);
    let owner_owned = Arc::clone(&owner);
    let api = model.api.clone();
    let provider_id = record.provider_id.clone();
    let model_id = model.id.clone();

    // The plugin call must not hold `prod`: forwarding happens on this task, so
    // a slow consumer cannot stall the plugin thread inside an FFI call.
    let request_model_id = model_id.clone();
    let join = runtime.spawn_blocking(move || {
        // Cancellation detaches this task; keep the library and host-side
        // context alive until the FFI call and all plugin-owned frees finish.
        let _lifetime = lifetime;
        run_streaming_provider_request(
            &snapshot_owned,
            &record_owned,
            stream_fn,
            &bridge,
            &owner_owned,
            &request_json,
            &request_model_id,
        )
    });

    // The mapper lives on THIS side: the callback only forwards raw chunks, so a
    // slow consumer cannot block the plugin's thread inside an FFI call, and
    // there is exactly one writer to the event producer.
    let mut mapper = StreamMapper::new(
        model.api.clone(),
        record.provider_id.clone(),
        model.id.clone(),
    );

    let mut cancelled_mid_stream = false;
    // Forward every chunk, in arrival order, until the channel closes (the
    // blocking task dropping its sender). A chunk the mapper parks behind the
    // host-emitted `Start` is drained on the spot, so ordering holds without a
    // second queue. The cancel arm can fire at any await point — including on the
    // very first `recv`, which is why it is selected rather than checked in the
    // loop body.
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                // Latch first: the plugin may be mid-generation and its next
                // poll of `is_cancelled` must see this, not a later race.
                control.cancel();
                cancelled_mid_stream = true;
                break;
            }
            chunk = rx.recv() => match chunk {
                Some(chunk) => {
                    if let Some(event) = mapper.apply(chunk) {
                        prod.push(event);
                    }
                    while let Some(deferred) = mapper.take_deferred() {
                        if let Some(event) = mapper.apply(deferred) {
                            prod.push(event);
                        }
                    }
                }
                // The plugin finished (or its task ended). Stop forwarding: the
                // terminal event is pushed below, after the join resolves.
                None => break,
            },
        }
    }

    // Normal completion has already drained the channel. Cancellation discards
    // queued chunks so no more content is forwarded after the cancel signal.
    rx.close();
    let mut message = if cancelled_mid_stream {
        // Dropping the join handle detaches the blocking task; waiting here
        // would let an uncooperative plugin hold the entire agent run open.
        mapper.aborted_message()
    } else {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                control.cancel();
                mapper.aborted_message()
            }
            outcome = join => match outcome {
                Ok(message) => message,
                Err(join_err) => AssistantMessage::terminal(
                    api,
                    provider_id,
                    model_id,
                    rpi_ai::types::StopReason::Error,
                    format!("plugin provider task failed: {join_err}"),
                    0,
                ),
            },
        }
    };
    if message.provider.is_empty() {
        message.provider = record.provider_id.clone();
    }
    if message.model.is_empty() {
        message.model = model.id.clone();
    }
    for event in mapper.chunks_then_message(&message) {
        prod.push(event);
    }
    message
}

/// The terminal result of driving a streaming provider: the plugin's own final
/// message, which becomes the `Done`/`Error` payload.
type StreamingProviderOutcome = AssistantMessage;

/// Select the legacy or protected streaming ABI for one request.
#[derive(Clone, Copy)]
enum StreamProviderCallback {
    Legacy(rpi_plugin_sdk::ProviderStreamRequestFn),
    Protected(rpi_plugin_sdk::ProtectedProviderStreamRequestFn),
}

/// Cancellation latched from `options.signal` for the plugin's synchronous
/// polling and blocking waits. The condition variable wakes every waiter when
/// the async driver observes cancellation.
#[derive(Default)]
struct ProviderControlState {
    cancelled: AtomicBool,
    wait_lock: Mutex<()>,
    wake: Condvar,
}

impl ProviderControlState {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Latch cancellation. Idempotent and thread-safe; the plugin may observe it
    /// at any time after this returns.
    fn cancel(&self) {
        // Serialize with the waiter's predicate check to avoid a lost wakeup.
        let _guard = self.wait_lock.lock().unwrap_or_else(|err| err.into_inner());
        self.cancelled.store(true, Ordering::SeqCst);
        self.wake.notify_all();
    }

    /// Block up to `timeout_ms` for cancellation. `0` is a non-blocking poll.
    /// Runs on the plugin's request thread and only touches this struct, so it
    /// cannot deadlock on any host lock.
    fn wait(&self, timeout_ms: u32) -> bool {
        if self.is_cancelled() {
            return true;
        }
        if timeout_ms == 0 {
            return false;
        }
        let guard = self.wait_lock.lock().unwrap_or_else(|err| err.into_inner());
        let _wait = self
            .wake
            .wait_timeout_while(guard, Duration::from_millis(u64::from(timeout_ms)), |_| {
                !self.is_cancelled()
            })
            .unwrap_or_else(|err| err.into_inner());
        self.is_cancelled()
    }
}

/// The owner of the control state, kept alive for the whole request. Handing the
/// plugin a pointer to an `Arc`-backed allocation (rather than a stack address)
/// means the address stays valid even if the request task is moved by the
/// blocking pool before it calls the plugin.
struct ProviderControlOwner {
    state: Arc<ProviderControlState>,
    secret: Option<(RequestSecret, bool)>,
    headers: Vec<(u32, String, String)>,
}

/// Copy request header pairs into the protected callback owner.
///
/// # Parameters
/// - `model`: Selected model containing model-level headers.
/// - `opts`: Per-request options containing request-level headers.
///
/// # Returns
/// Header name/value pairs tagged with their source.
fn protected_headers(model: &Model, opts: &SimpleStreamOptions) -> Vec<(u32, String, String)> {
    let model_headers = model.headers.iter().flat_map(|headers| {
        headers.iter().map(|(name, value)| {
            (
                rpi_plugin_sdk::ProviderHeaderSource::Model as u32,
                name.clone(),
                value.clone(),
            )
        })
    });
    let option_headers = opts.headers.iter().flat_map(|headers| {
        headers.iter().map(|(name, value)| {
            (
                rpi_plugin_sdk::ProviderHeaderSource::RequestOptions as u32,
                name.clone(),
                value.clone(),
            )
        })
    });
    model_headers.chain(option_headers).collect()
}

/// Build a size-versioned protected control value for one plugin call.
///
/// # Parameters
/// - `owner`: Host-owned callback state that outlives the plugin call.
///
/// # Returns
/// Protected controls with only callbacks available for present inputs.
fn protected_control(owner: &ProviderControlOwner) -> rpi_plugin_sdk::ProtectedProviderControl {
    rpi_plugin_sdk::ProtectedProviderControl {
        struct_size: std::mem::size_of::<rpi_plugin_sdk::ProtectedProviderControl>() as u32,
        stream: rpi_plugin_sdk::ProviderStreamControl {
            is_cancelled: control_is_cancelled,
            wait_cancelled_ms: control_wait_cancelled_ms,
            user_data: owner as *const ProviderControlOwner as *mut c_void,
        },
        with_request_secret: owner
            .secret
            .as_ref()
            .map(|_| control_with_request_secret as rpi_plugin_sdk::ProviderSecretFn),
        request_secret_bearer: owner.secret.as_ref().is_some_and(|(_, bearer)| *bearer),
        with_request_headers: (!owner.headers.is_empty())
            .then_some(control_with_request_headers as rpi_plugin_sdk::ProviderHeadersFn),
        user_data: owner as *const ProviderControlOwner as *mut c_void,
    }
}

/// Expose request headers only during synchronous protected consumption.
extern "C" fn control_with_request_headers(
    consumer: rpi_plugin_sdk::ProviderHeaderConsumerFn,
    consumer_user_data: *mut c_void,
    control_user_data: *mut c_void,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if control_user_data.is_null() {
            return 1;
        }
        // SAFETY: the owner remains alive for the complete plugin request call.
        let owner = unsafe { &*(control_user_data as *const ProviderControlOwner) };
        for (source, name, value) in &owner.headers {
            if consumer(
                *source,
                StbStringRef::from_str(name),
                StbStringRef::from_str(value),
                consumer_user_data,
            ) != 0
            {
                return 1;
            }
        }
        0
    }))
    .unwrap_or(1)
}

/// Expose the request credential only during a synchronous plugin callback.
extern "C" fn control_with_request_secret(
    consumer: rpi_plugin_sdk::ProviderSecretConsumerFn,
    consumer_user_data: *mut c_void,
    control_user_data: *mut c_void,
) -> i32 {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if control_user_data.is_null() {
            return 1;
        }
        // SAFETY: the owner remains alive for the complete plugin request call.
        let owner = unsafe { &*(control_user_data as *const ProviderControlOwner) };
        let Some((secret, _)) = &owner.secret else {
            return 1;
        };
        let mut result = 1;
        secret.with_secret(&mut |value| {
            result = consumer(StbStringRef::from_str(value), consumer_user_data);
        });
        result
    }));
    outcome.unwrap_or(1)
}

/// The host's two plugin-facing control trampolines.
///
/// Both take the `user_data` the plugin got from
/// [`rpi_plugin_sdk::ProviderStreamControl`], which is **always** the live
/// `ProviderControlOwner` on the v2 path — so an early return for `user_data ==
/// null` is unreachable from a well-formed plugin and exists only as the
/// defensive floor against a forged pointer.
///
/// `is_cancelled`:
///
/// `chunk_kind` distinguishes content from structural chunks: a cancelled
/// request may still deliver structural frames that close the message the user
/// already saw, but has nobody left to receive model output.
extern "C" fn control_is_cancelled(user_data: *mut c_void, chunk_kind: u32) -> i32 {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            // Should not happen: the host always passes a live owner. Report
            // "not cancelled" rather than aborting the plugin's request.
            tracing::error!("provider control callback missing its owner");
            return 0;
        }
        // SAFETY: the owner is heap-allocated and kept alive by the request's
        // `Arc<ProviderControlOwner>` for the whole plugin call.
        let owner = unsafe { &*(user_data as *const ProviderControlOwner) };
        if chunk_kind == rpi_plugin_sdk::ProviderChunkKind::Structural as u32 {
            return 0;
        }
        i32::from(owner.state.is_cancelled())
    }));
    outcome.unwrap_or(0)
}

/// `wait_cancelled_ms`: blocks up to `timeout_ms` on the plugin's own thread and
/// returns nonzero as soon as cancellation is observed.
extern "C" fn control_wait_cancelled_ms(user_data: *mut c_void, timeout_ms: u32) -> i32 {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            tracing::error!("provider control callback missing its owner");
            return 0;
        }
        // SAFETY: as in `control_is_cancelled` — the owner outlives the call.
        let owner = unsafe { &*(user_data as *const ProviderControlOwner) };
        i32::from(owner.state.wait(timeout_ms))
    }));
    outcome.unwrap_or(0)
}

/// The callback state an emitted chunk is funneled through. Owned by the host,
/// pointed at by the plugin's `emit_user_data`, and — because it holds the only
/// sender of the chunk channel — the channel's lifetime marker: when the
/// blocking task drops it, the consumer's `recv()` returns `None`.
struct StreamEmitBridge {
    tx: tokio::sync::mpsc::UnboundedSender<StreamChunk>,
    /// The plugin's own `plugin_free_string`: every emitted chunk arrives as a
    /// **plugin-owned** allocation, so only the plugin's deallocator may reclaim
    /// it (the host's would corrupt the plugin's heap).
    plugin_free_string: rpi_plugin_sdk::FreeStringFn,
    /// Latched cancellation shared with the plugin's control callbacks.
    control: Option<Arc<ProviderControlState>>,
}

/// One emitted chunk, already copied out of the plugin's allocation and safely
/// reclaimed through the plugin's `free_string`.
type StreamChunk = serde_json::Value;

/// The host's per-chunk callback. Invoked **synchronously inside the plugin's
/// request fn**, on the blocking-pool thread.
///
/// Ownership: `event_json` is plugin-produced, so it is reclaimed here with the
/// plugin's registered `plugin_free_string` — never with the host's allocator.
/// A missing context, a malformed envelope, or a panic in the host's own parsing
/// is reported as a nonzero rc; the plugin's string is still reclaimed exactly
/// once on every path, so a rejected chunk cannot leak.
extern "C" fn stream_emit_trampoline(event_json: StbString, user_data: *mut c_void) -> i32 {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            // Invalid ABI invocation: there is no bridge to tell us which
            // deallocator owns this allocation. Never guess — logging is all we
            // can safely do without risking a cross-allocator free.
            tracing::error!("plugin provider emit callback missing its context");
            return 2;
        }
        // SAFETY: the blocking task owns this bridge for the entire plugin
        // request, including after the async driver returns on cancellation.
        let bridge = unsafe { &*(user_data as *const StreamEmitBridge) };
        // SAFETY: the plugin transfers a live, plugin-owned allocation exactly
        // once; `plugin_free_string` is that plugin's deallocator.
        let owned = unsafe { OwnedString::from_raw(event_json, Some(bridge.plugin_free_string)) };
        let text = owned.to_string_lossy();
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(chunk) if chunk.is_object() => {
                // A cancelled run still needs the structural frames that close
                // the message it already showed, but has no consumer left for
                // model output. Rejecting content here is what tells a plugin to
                // stop producing instead of burning a full generation.
                let is_content = matches!(
                    chunk.get("type").and_then(serde_json::Value::as_str),
                    Some("text_delta") | Some("thinking_delta") | Some("toolcall_delta")
                );
                if bridge
                    .control
                    .as_ref()
                    .is_some_and(|state| state.is_cancelled())
                {
                    // After the aborted terminal, acknowledge cleanup frames
                    // without enqueueing events that no longer have a consumer.
                    return i32::from(is_content);
                }
                if bridge.tx.send(chunk).is_err() {
                    // The consumer went away (cancelled run); nothing to deliver.
                    return 1;
                }
                0
            }
            Ok(_) => {
                tracing::warn!("plugin provider emitted a non-object chunk — dropped");
                1
            }
            Err(error) => {
                tracing::warn!(%error, "plugin provider emitted unparseable JSON — dropped");
                1
            }
        }
    }));
    match outcome {
        Ok(rc) => rc,
        Err(_) => {
            tracing::error!("plugin provider emit callback panicked — chunk dropped");
            2
        }
    }
}

/// Run the streaming request fn inside `catch_unwind`, returning the plugin's
/// final message (or an error terminal). Mirrors
/// [`run_provider_request_inner`]'s contract: a null/empty `out`, a nonzero rc,
/// or an unparseable response all become an error terminal rather than a crash.
fn run_streaming_provider_request(
    snapshot: &Arc<RegistrySnapshot>,
    record: &RegisteredProvider,
    stream_fn: StreamProviderCallback,
    bridge: &StreamEmitBridge,
    owner: &ProviderControlOwner,
    request_json: &str,
    model_id: &str,
) -> StreamingProviderOutcome {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !snapshot.is_active() {
            return Err("extensions provider registry is stale".to_string());
        }
        let req_ref = StbStringRef::from_str(request_json);
        let mut out = StbString::empty();
        let relay = bridge as *const StreamEmitBridge as *mut std::ffi::c_void;
        // `user_data` for the control callbacks is the owner's own address: the
        // trampolines cast it back, and the `Arc` keeps it alive for the call.
        let control = rpi_plugin_sdk::ProviderStreamControl {
            is_cancelled: control_is_cancelled,
            wait_cancelled_ms: control_wait_cancelled_ms,
            user_data: owner as *const ProviderControlOwner as *mut std::ffi::c_void,
        };
        let protected = protected_control(owner);
        let rc = match stream_fn {
            StreamProviderCallback::Legacy(stream_fn) => stream_fn(
                req_ref,
                Some(stream_emit_trampoline),
                relay,
                &control,
                None,
                std::ptr::null_mut(),
                &mut out,
                record.user_data,
            ),
            StreamProviderCallback::Protected(stream_fn) => stream_fn(
                req_ref,
                Some(stream_emit_trampoline),
                relay,
                &protected,
                None,
                std::ptr::null_mut(),
                &mut out,
                record.user_data,
            ),
        };
        // SAFETY: the callback transfers its live output to its registered
        // producer, on every rc path (including an empty slot).
        let out = unsafe {
            rpi_plugin_sdk::callbacks::OwnedString::from_raw(out, Some(record.plugin_free_string))
        };
        if rc != 0 {
            // A cancelled run makes the plugin return early; that is the
            // expected outcome, not a provider failure. The driver turns it into
            // an `Aborted` terminal, so it is reported as a cancellation here.
            if bridge
                .control
                .as_ref()
                .is_some_and(|state| state.is_cancelled())
            {
                return Err(CANCELLED_MARKER.to_string());
            }
            return Err(format!(
                "plugin provider {} streaming request returned error code {rc}",
                record.provider_id
            ));
        }
        let response_text = out.to_string_lossy();
        serde_json::from_str::<AssistantMessage>(&response_text).map_err(|err| {
            format!(
                "plugin provider {} returned unparseable response: {err}",
                record.provider_id
            )
        })
    }));

    match outcome {
        Ok(Ok(mut message)) => {
            if message.provider.is_empty() {
                message.provider = record.provider_id.clone();
            }
            if message.model.is_empty() {
                message.model = model_id.to_string();
            }
            message
        }
        Ok(Err(msg)) if msg == CANCELLED_MARKER => cancelled_message(record),
        Ok(Err(msg)) => error_message(record, &msg),
        Err(_) => {
            tracing::error!(
                "plugin provider {} panicked — refusing to unwind across FFI",
                record.provider_id
            );
            error_message(
                record,
                &format!(
                    "plugin provider {} panicked during request",
                    record.provider_id
                ),
            )
        }
    }
}

/// Sentinel the blocking request runner returns so the driver can tell "the run
/// was cancelled" from "the plugin failed". It cannot collide with a real plugin
/// error: plugin-side failures are formatted as `plugin provider {id} …`, and
/// this marker starts with a NUL no plugin-produced text goes through.
const CANCELLED_MARKER: &str = "\u{0}rpi:cancelled";

/// The terminal message for a request whose caller went away. `StopReason::Aborted`
/// is what the built-in providers use, and the crate's `push_terminal` maps it to
/// `AssistantMessageEvent::Error { reason: Aborted }` — so a plugin provider is
/// indistinguishable from a built-in one in the aborted case.
fn cancelled_message(record: &RegisteredProvider) -> AssistantMessage {
    AssistantMessage::terminal(
        api_from_style(&record.api_style),
        record.provider_id.clone(),
        String::new(),
        rpi_ai::types::StopReason::Aborted,
        "plugin provider request was cancelled",
        0,
    )
}

/// An error terminal for a plugin provider. The api string is decoded exactly
/// like a deserialized `Api` (an unknown style becomes `Api::Other`), so an error
/// message from a plugin provider carries the same `api` a success would.
fn error_message(record: &RegisteredProvider, message: &str) -> AssistantMessage {
    AssistantMessage::terminal(
        api_from_style(&record.api_style),
        record.provider_id.clone(),
        String::new(),
        rpi_ai::types::StopReason::Error,
        message.to_string(),
        0,
    )
}

/// Decode the registration's `api_style` into an [`Api`](rpi_ai::types::Api).
/// Mirrors `Api`'s own `Deserialize` (including the `Other(..)` fallback) without
/// needing a throwaway serde pass.
fn api_from_style(style: &str) -> rpi_ai::types::Api {
    serde_json::from_value::<rpi_ai::types::Api>(serde_json::Value::String(style.to_string()))
        .unwrap_or_else(|_| rpi_ai::types::Api::Other(style.to_string()))
}

/// Build the provider request envelope shared by one-shot and streaming paths.
///
/// `providerOptions` is a separately versioned, provider-specific configuration
/// object. It is omitted when empty and never enters serialized public model
/// metadata.
fn request_envelope_json(
    model: &Model,
    ctx: &Context,
    opts: &SimpleStreamOptions,
) -> Result<String, String> {
    let mut model = model.clone();
    model.headers = None;
    let mut request = serde_json::Map::from_iter([
        (
            "model".to_owned(),
            serde_json::to_value(&model).unwrap_or(serde_json::Value::Null),
        ),
        (
            "context".to_owned(),
            serde_json::to_value(ctx).unwrap_or(serde_json::Value::Null),
        ),
        ("options".to_owned(), options_json(opts)),
    ]);
    if !model.provider_options.is_empty() {
        request.insert(
            "providerOptions".to_owned(),
            serde_json::json!({
                "version": 1,
                "values": &model.provider_options,
            }),
        );
    }
    Ok(serde_json::Value::Object(request).to_string())
}

/// Live accumulator for the chunks a streaming provider emits.
///
/// The mapper owns the growing `AssistantMessage` the deltas are folded into, so
/// every `*_delta` event can carry a `partial` snapshot without the plugin having
/// to send one — the same shape the built-in providers produce (an
/// `Arc<AssistantMessage>` per event, cloned once per delta).
struct StreamMapper {
    output: AssistantMessage,
    /// Whether the stream has been opened by a `Start` event. The host opens it
    /// lazily on the first chunk so a plugin that emits nothing is never
    /// preceded by a stray `Start`.
    started: bool,
    /// Whether the terminal event has been produced; a post-terminal chunk is
    /// dropped (mirrors the provider-side "no events after terminal" rule).
    terminal: bool,
    /// Raw tool-call JSON accumulated per content index, for the same reason the
    /// Anthropic mapper keeps a `partial_json` buffer: the `arguments` object a
    /// `ToolCallEnd` carries is the *authoritative* parse, and intermediate
    /// parses must never produce a value the cumulative JSON cannot justify.
    tool_json: std::collections::BTreeMap<usize, String>,
    /// A chunk parked because the host's `Start` had to be delivered first. Set
    /// when a chunk arrives before the stream is open; re-queued by the driver on
    /// the next iteration so no delta is lost or reordered.
    deferred: Option<serde_json::Value>,
    /// The plugin's own final message, once it has returned. `Some` only after
    /// [`Self::chunks_then_message`].
    final_message: Option<AssistantMessage>,
}

impl StreamMapper {
    fn new(api: rpi_ai::types::Api, provider: String, model: String) -> Self {
        Self {
            output: AssistantMessage::empty(api, provider, model, 0),
            started: false,
            terminal: false,
            tool_json: std::collections::BTreeMap::new(),
            deferred: None,
            final_message: None,
        }
    }

    /// Take the chunk parked behind the host-emitted `Start`.
    fn take_deferred(&mut self) -> Option<serde_json::Value> {
        self.deferred.take()
    }

    fn ensure_started(&mut self) -> Option<AssistantMessageEvent> {
        if self.started {
            return None;
        }
        self.started = true;
        Some(AssistantMessageEvent::Start {
            partial: Arc::new(self.output.clone()),
        })
    }

    fn partial(&self) -> Arc<AssistantMessage> {
        Arc::new(self.output.clone())
    }

    fn aborted_message(&self) -> AssistantMessage {
        let mut message = self.output.clone();
        message.stop_reason = rpi_ai::types::StopReason::Aborted;
        message.error_message = Some("plugin provider request was cancelled".into());
        message
    }

    /// A supplied delta snapshot already includes that delta. Replace the
    /// accumulator without appending the same text/arguments a second time.
    fn apply_partial(&mut self, chunk: &serde_json::Value) -> bool {
        match chunk
            .get("partial")
            .cloned()
            .map(serde_json::from_value::<AssistantMessage>)
        {
            Some(Ok(message)) => {
                self.output = message;
                true
            }
            Some(Err(error)) => {
                tracing::warn!(%error, "ignoring malformed plugin provider partial snapshot");
                false
            }
            None => false,
        }
    }

    /// Fold one plugin chunk into the accumulator and return the event it maps
    /// to. `None` means "nothing to emit" (an unknown `type`, or a malformed
    /// chunk the plugin is allowed to get wrong without killing the run).
    fn apply(&mut self, chunk: serde_json::Value) -> Option<AssistantMessageEvent> {
        if self.terminal {
            tracing::debug!("dropping a plugin provider chunk that follows a terminal event");
            return None;
        }
        let mut chunk = chunk;
        let kind = chunk.get("type").and_then(serde_json::Value::as_str)?;

        // A chunk that opens or continues a block implies the stream is open.
        // `Start` is emitted first so the consumer's `MessageStart` never lands
        // after a delta (the agent loop and the frame encoder both require the
        // ordering). The chunk is parked and re-applied by `drive_...`'s next
        // iteration, so no delta is lost or reordered.
        if !self.started {
            self.started = true;
            self.deferred = Some(std::mem::replace(&mut chunk, serde_json::Value::Null));
            return Some(AssistantMessageEvent::Start {
                partial: self.partial(),
            });
        }

        match kind {
            // `start` is not an event of its own: the host emits `Start` above
            // the first time it sees *any* chunk, so a duplicate is dropped.
            "start" => None,
            "usage" => {
                // Usage is not an event of its own: it patches the accumulating
                // message so the terminal one reports it. A plugin that sends no
                // usage keeps the zeroed default.
                if let Some(usage) = chunk.get("usage") {
                    if let Ok(parsed) =
                        serde_json::from_value::<rpi_ai::types::Usage>(usage.clone())
                    {
                        self.output.usage = parsed;
                    }
                }
                None
            }
            "text_start" => {
                let index = self.content_index(&chunk);
                self.open_block(index, Content::text(String::new()));
                Some(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: self.partial(),
                })
            }
            "text_delta" => {
                let index = self.content_index(&chunk);
                let delta = chunk_string(&chunk, "delta");
                if !self.apply_partial(&chunk) {
                    let block = self.ensure_text_block(index);
                    block.text.push_str(&delta);
                }
                Some(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta,
                    partial: self.partial(),
                })
            }
            "text_end" => {
                let index = self.content_index(&chunk);
                let block = self.ensure_text_block(index);
                let content = if chunk.get("content").is_some() {
                    chunk_string(&chunk, "content")
                } else {
                    block.text.clone()
                };
                block.text = content.clone();
                Some(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content,
                    partial: self.partial(),
                })
            }
            "thinking_start" => {
                let index = self.content_index(&chunk);
                self.open_block(index, Content::thinking(String::new()));
                Some(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: self.partial(),
                })
            }
            "thinking_delta" => {
                let index = self.content_index(&chunk);
                let delta = chunk_string(&chunk, "delta");
                if !self.apply_partial(&chunk) {
                    let block = self.ensure_thinking_block(index);
                    block.thinking.push_str(&delta);
                }
                Some(AssistantMessageEvent::ThinkingDelta {
                    content_index: index,
                    delta,
                    partial: self.partial(),
                })
            }
            "thinking_end" => {
                let index = self.content_index(&chunk);
                let block = self.ensure_thinking_block(index);
                let content = if chunk.get("content").is_some() {
                    chunk_string(&chunk, "content")
                } else {
                    block.thinking.clone()
                };
                block.thinking = content.clone();
                let redacted = chunk
                    .get("redacted")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                block.redacted = redacted;
                Some(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content,
                    partial: self.partial(),
                })
            }
            "toolcall_start" => {
                let index = self.content_index(&chunk);
                let id = chunk_string(&chunk, "id");
                let name = chunk_string(&chunk, "name");
                // The `arguments` snapshot at start is authoritative for the
                // arguments the block already has; the delta stream then extends
                // it. A plugin that sends only deltas starts from an empty
                // object, which is what the frame encoder expects.
                let arguments = chunk
                    .get("arguments")
                    .cloned()
                    .filter(serde_json::Value::is_object)
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                self.tool_json.insert(index, String::new());
                self.open_block(index, Content::tool_call(id, name, arguments));
                Some(AssistantMessageEvent::ToolCallStart {
                    content_index: index,
                    partial: self.partial(),
                })
            }
            "toolcall_delta" => {
                let index = self.content_index(&chunk);
                let delta = chunk_string(&chunk, "delta");
                let accumulated = self.tool_json.entry(index).or_default();
                accumulated.push_str(&delta);
                // Re-parse on every delta so the partial message renders live
                // arguments (the same invariant the Anthropic mapper holds).
                let parsed = rpi_ai::json_parse::parse_streaming_json(Some(accumulated));
                if !self.apply_partial(&chunk) {
                    match self.output.content.get_mut(index) {
                        Some(Content::ToolCall(call)) => call.arguments = parsed,
                        _ => {
                            let id = chunk_string(&chunk, "id");
                            let name = chunk_string(&chunk, "name");
                            self.ensure_tool_call_block(index, id, name);
                            if let Some(Content::ToolCall(call)) =
                                self.output.content.get_mut(index)
                            {
                                call.arguments = parsed;
                            }
                        }
                    }
                }
                Some(AssistantMessageEvent::ToolCallDelta {
                    content_index: index,
                    delta,
                    partial: self.partial(),
                })
            }
            "toolcall_end" => {
                let index = self.content_index(&chunk);
                let id = chunk_string(&chunk, "id");
                let name = chunk_string(&chunk, "name");
                let arguments = chunk
                    .get("arguments")
                    .cloned()
                    .filter(serde_json::Value::is_object)
                    .or_else(|| match self.output.content.get(index) {
                        Some(Content::ToolCall(call)) => Some(call.arguments.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                self.ensure_tool_call_block(index, id.clone(), name.clone());
                let call = match self.output.content.get_mut(index) {
                    Some(Content::ToolCall(call)) => {
                        if !id.is_empty() {
                            call.id = id.clone();
                        }
                        if !name.is_empty() {
                            call.name = name.clone();
                        }
                        call.arguments = arguments.clone();
                        call.clone()
                    }
                    _ => ToolCall {
                        kind: ToolCallType,
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        thought_signature: None,
                        namespace: None,
                    },
                };
                self.tool_json.remove(&index);
                Some(AssistantMessageEvent::ToolCallEnd {
                    content_index: index,
                    tool_call: call,
                    partial: self.partial(),
                })
            }
            other => {
                tracing::debug!(
                    chunk_type = other,
                    "ignoring an unknown plugin provider chunk"
                );
                None
            }
        }
    }

    fn content_index(&self, chunk: &serde_json::Value) -> usize {
        chunk
            .get("contentIndex")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(0)
    }

    /// Append a freshly-opened block at `index`, padding any gap with empty text
    /// blocks so the index the plugin named is the index the message has. A
    /// plugin that names indices out of order is a plugin bug, but it must not
    /// corrupt the message the host persists.
    fn open_block(&mut self, index: usize, content: Content) {
        while self.output.content.len() <= index {
            self.output.content.push(Content::text(String::new()));
        }
        self.output.content[index] = content;
    }

    fn ensure_text_block(&mut self, index: usize) -> &mut rpi_ai::types::TextContent {
        self.ensure_block(index, || Content::text(String::new()));
        match self.output.content.get_mut(index) {
            Some(Content::Text(block)) => block,
            other => unreachable!("ensure_text_block left a non-text block: {other:?}"),
        }
    }

    fn ensure_thinking_block(&mut self, index: usize) -> &mut rpi_ai::types::ThinkingContent {
        self.ensure_block(index, || Content::thinking(String::new()));
        match self.output.content.get_mut(index) {
            Some(Content::Thinking(block)) => block,
            other => unreachable!("ensure_thinking_block left a non-thinking block: {other:?}"),
        }
    }

    fn ensure_tool_call_block(&mut self, index: usize, id: String, name: String) {
        self.ensure_block(index, || {
            Content::tool_call(
                id.clone(),
                name.clone(),
                serde_json::Value::Object(serde_json::Map::new()),
            )
        });
    }

    fn ensure_block(&mut self, index: usize, make: impl FnOnce() -> Content) {
        if self.output.content.len() <= index {
            // No `*_start` opened this block: the plugin skipped it. Repair the
            // message (the deltas still land) but say so — a missing start is a
            // contract violation the plugin author should see in the logs rather
            // than a silently different message shape.
            tracing::warn!(
                content_index = index,
                "plugin provider sent a block delta without its start chunk — synthesizing the block"
            );
            while self.output.content.len() < index {
                self.output.content.push(Content::text(String::new()));
            }
            self.output.content.push(make());
        }
    }

    /// Events that must precede the terminal event for `message`: whatever the
    /// mapper has not yet delivered (at minimum a `Start`, so a plugin that
    /// emitted nothing still produces a well-formed stream).
    /// `message` is the plugin's final payload on completion, or the accumulated
    /// partial with an Aborted stop reason when the caller cancels.
    fn chunks_then_message(&mut self, message: &AssistantMessage) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        if let Some(start) = self.ensure_started() {
            events.push(start);
        }
        self.terminal = true;
        self.final_message = Some(message.clone());
        events
    }

    /// The accumulated message, for tests and diagnostics.
    #[cfg(test)]
    fn accumulated(&self) -> &AssistantMessage {
        &self.output
    }
}

/// Read a string field from a chunk, defaulting to empty.
fn chunk_string(chunk: &serde_json::Value, field: &str) -> String {
    chunk
        .get(field)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ExtensionRegistry;

    /// Provider model options reach extension providers through the versioned
    /// request envelope while staying out of public model serialization.
    #[test]
    fn provider_options_use_versioned_extension_envelope() {
        let mut model = Model::new(
            "m",
            "Model",
            rpi_ai::types::Api::OpenaiCompletions,
            "provider",
            "https://example.test",
        );
        let empty_request: serde_json::Value = serde_json::from_str(
            &request_envelope_json(
                &model,
                &Context::new(vec![]),
                &SimpleStreamOptions::default(),
            )
            .expect("safe empty provider request envelope"),
        )
        .expect("empty provider request envelope");
        assert!(empty_request.get("providerOptions").is_none());

        model
            .provider_options
            .insert("reasoning_effort".to_string(), serde_json::json!("high"));
        model.provider_options.insert(
            "reasoning_budget".to_string(),
            serde_json::json!("opaque-secret-sentinel"),
        );

        let public_model = serde_json::to_value(&model).expect("public model JSON");
        let request: serde_json::Value = serde_json::from_str(
            &request_envelope_json(
                &model,
                &Context::new(vec![]),
                &SimpleStreamOptions::default(),
            )
            .expect("safe provider options request envelope"),
        )
        .expect("provider request envelope");

        assert!(public_model.get("providerOptions").is_none());
        assert!(!format!("{model:?}").contains("opaque-secret-sentinel"));
        assert_eq!(request["providerOptions"]["version"], 1);
        assert_eq!(
            request["providerOptions"]["values"]["reasoning_effort"],
            "high"
        );
        assert_eq!(
            request["providerOptions"]["values"]["reasoning_budget"],
            "opaque-secret-sentinel"
        );
    }

    /// A callback state used to verify manifest-gated provider requests.
    #[derive(Default)]
    struct ProviderOptionsProbe {
        /// Number of times the plugin callback was invoked.
        calls: std::sync::atomic::AtomicUsize,
        /// Request envelope received by the callback.
        request: Mutex<Option<String>>,
    }

    /// Captures the request envelope and returns a valid assistant message.
    extern "C" fn capture_provider_options_request(
        request: StbStringRef,
        out: *mut StbString,
        user_data: *mut c_void,
    ) -> i32 {
        // SAFETY: the test keeps the probe alive until the provider stream ends.
        let probe = unsafe { &*user_data.cast::<ProviderOptionsProbe>() };
        // SAFETY: the host passes a valid borrowed request for this callback.
        let request = unsafe { request.as_str() };
        probe.calls.fetch_add(1, Ordering::AcqRel);
        *probe.request.lock().unwrap() = Some(request.to_owned());
        let response = r#"{"role":"assistant","content":[{"type":"text","text":"ok"}],"api":"faux","provider":"pluggy","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":0}"#;
        // SAFETY: the host supplies the writable output slot for this callback.
        unsafe {
            *out = StbString::from_string(response.to_owned());
        }
        0
    }

    /// Creates a registered provider snapshot with optional v1 declaration.
    fn provider_options_snapshot(version: Option<u32>) -> Arc<RegistrySnapshot> {
        let mut registry = ExtensionRegistry::new();
        let mut declaration = serde_json::json!({
            "providers": [{
                "id": "pluggy",
                "models": [Model::new(
                    "m",
                    "Model",
                    rpi_ai::types::Api::Faux,
                    "pluggy",
                    "https://example.test"
                )]
            }]
        });
        if let Some(version) = version {
            declaration["providers"][0]["providerOptionsVersion"] = serde_json::json!(version);
        }
        assert!(registry.apply_declaration_json(&declaration.to_string()));
        registry.register_provider(RegisteredProvider {
            provider_id: "pluggy".to_owned(),
            base_url: "https://example.test".to_owned(),
            api_style: "faux".to_owned(),
            request_fn: capture_provider_options_request,
            protected_request_fn: None,
            protected_stream_fn: None,
            stream_fn: None,
            plugin_free_string: stub_free,
            user_data: std::ptr::null_mut(),
        });
        Arc::new(registry.snapshot())
    }

    /// Enforces opt-in before callback and forwards options after v1 opt-in.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provider_options_require_manifest_opt_in_at_runtime() {
        let runtime = Handle::current();
        let context = Context::new(vec![]);
        let options = SimpleStreamOptions::default();
        let model = Model::new(
            "m",
            "Model",
            rpi_ai::types::Api::Faux,
            "pluggy",
            "https://example.test",
        );
        let legacy_probe = Arc::new(ProviderOptionsProbe::default());
        let legacy_provider = PluggableProvider::new(
            RegisteredProvider {
                provider_id: "pluggy".to_owned(),
                base_url: "https://example.test".to_owned(),
                api_style: "faux".to_owned(),
                request_fn: capture_provider_options_request,
                protected_request_fn: None,
                protected_stream_fn: None,
                stream_fn: None,
                plugin_free_string: stub_free,
                user_data: Arc::as_ptr(&legacy_probe) as *mut c_void,
            },
            provider_options_snapshot(None),
            PluginKeepalive::empty(),
            runtime.clone(),
        );
        let mut legacy_empty_stream = legacy_provider
            .stream_simple(&model, &context, &options)
            .await;
        let legacy_empty_terminal = legacy_empty_stream
            .next()
            .await
            .expect("legacy empty success terminal");
        assert!(matches!(
            legacy_empty_terminal,
            AssistantMessageEvent::Done { .. }
        ));
        assert_eq!(legacy_probe.calls.load(Ordering::Acquire), 1);
        let legacy_empty_request = legacy_probe
            .request
            .lock()
            .unwrap()
            .clone()
            .expect("legacy empty request envelope");
        let legacy_empty_request: serde_json::Value =
            serde_json::from_str(&legacy_empty_request).expect("legacy request JSON");
        assert!(legacy_empty_request.get("providerOptions").is_none());

        let mut model_with_options = model.clone();
        model_with_options
            .provider_options
            .insert("reasoning_effort".to_owned(), serde_json::json!("high"));
        let mut legacy_stream = legacy_provider
            .stream_simple(&model_with_options, &context, &options)
            .await;
        let legacy_terminal = legacy_stream.next().await.expect("error terminal");
        assert!(matches!(
            legacy_terminal,
            AssistantMessageEvent::Error { .. }
        ));
        assert_eq!(legacy_probe.calls.load(Ordering::Acquire), 1);

        let opted_probe = Arc::new(ProviderOptionsProbe::default());
        let opted_provider = PluggableProvider::new(
            RegisteredProvider {
                provider_id: "pluggy".to_owned(),
                base_url: "https://example.test".to_owned(),
                api_style: "faux".to_owned(),
                request_fn: capture_provider_options_request,
                protected_request_fn: None,
                protected_stream_fn: None,
                stream_fn: None,
                plugin_free_string: stub_free,
                user_data: Arc::as_ptr(&opted_probe) as *mut c_void,
            },
            provider_options_snapshot(Some(1)),
            PluginKeepalive::empty(),
            runtime,
        );
        let mut opted_stream = opted_provider
            .stream_simple(&model_with_options, &context, &options)
            .await;
        let opted_terminal = opted_stream.next().await.expect("success terminal");
        assert!(matches!(opted_terminal, AssistantMessageEvent::Done { .. }));
        assert_eq!(opted_probe.calls.load(Ordering::Acquire), 1);
        let request = opted_probe
            .request
            .lock()
            .unwrap()
            .clone()
            .expect("plugin request envelope");
        let request: serde_json::Value =
            serde_json::from_str(&request).expect("valid provider request JSON");
        assert_eq!(request["providerOptions"]["version"], 1);
        assert_eq!(
            request["providerOptions"]["values"]["reasoning_effort"],
            "high"
        );
    }

    rpi_plugin_sdk::string_handler! {
        /// A stub `ProviderRequestFn` that writes a minimal assistant-message JSON
        /// into `out`. Mirrors the shape a real plugin's request fn returns.
        fn ok_request_fn(_req, out, _ud) {
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"hi"}],"api":"faux","provider":"pluggy","model":"","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":0}"#;
        i32::from(!out.write(json.to_string()))
    } }

    /// A free fn matching the stub's allocation (`StbString::from_string` ⇒ a
    /// `Box<[u8]>` reclaimed by reconstructing the slice).
    use rpi_plugin_sdk::callbacks::free_string as stub_free;

    /// A v2 "provider" whose synchronous request body blocks the way a real one
    /// would (it waits for cancellation after an initial burst) and reports what
    /// it saw. Used to drive the whole cancellation path in-process, without the
    /// cdylib, so the decision logic is covered by a fast unit test too.
    mod cancel_provider {
        use super::*;
        use rpi_plugin_sdk::{ProviderStreamCompletionFn, ProviderStreamControl};
        use std::sync::atomic::AtomicUsize;

        pub static EMITTED: AtomicUsize = AtomicUsize::new(0);
        pub static OBSERVED: AtomicUsize = AtomicUsize::new(0);

        pub extern "C" fn request_v2(
            _req: StbStringRef,
            emit: Option<rpi_plugin_sdk::ProviderStreamEmitFn>,
            emit_user_data: *mut c_void,
            control: *const ProviderStreamControl,
            _completion: Option<ProviderStreamCompletionFn>,
            _completion_user_data: *mut c_void,
            out: *mut StbString,
            _ud: *mut c_void,
        ) -> i32 {
            let control = unsafe { *control };
            if let Some(emit) = emit {
                let start = r#"{"type":"text_start","contentIndex":0}"#;
                let _ = emit(StbString::from_string(start.to_string()), emit_user_data);
                for i in 0..50 {
                    // Block on the host's control surface, exactly as a provider
                    // parked in an SSE read would.
                    if (control.wait_cancelled_ms)(control.user_data, 20) != 0 {
                        OBSERVED.fetch_add(1, Ordering::SeqCst);
                        return 1;
                    }
                    let chunk =
                        format!(r#"{{"type":"text_delta","contentIndex":0,"delta":"{i},"}}"#);
                    if emit(StbString::from_string(chunk), emit_user_data) != 0 {
                        OBSERVED.fetch_add(1, Ordering::SeqCst);
                        EMITTED.fetch_add(1, Ordering::SeqCst);
                        return 1;
                    }
                    EMITTED.fetch_add(1, Ordering::SeqCst);
                }
            }
            let json = r#"{"role":"assistant","content":[{"type":"text","text":"completed"}],"api":"faux","provider":"cancel-probe","model":"m","usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":0}"#;
            unsafe {
                *out = StbString::from_string(json.to_string());
            }
            0
        }
    }

    /// `run_provider_request_inner` round-trips a nonzero-rc plugin response to
    /// `Err`, and a valid-message response to `Ok` (with provider stamped from
    /// the record when the plugin omits it).
    #[test]
    fn request_fn_round_trip_ok_and_err() {
        let mut registry = ExtensionRegistry::new();
        registry.register_provider(RegisteredProvider {
            provider_id: "pluggy".to_string(),
            base_url: "https://example".to_string(),
            api_style: "anthropic-messages".to_string(),
            request_fn: ok_request_fn,
            protected_request_fn: None,
            protected_stream_fn: None,
            stream_fn: None,
            plugin_free_string: stub_free,
            user_data: std::ptr::null_mut(),
        });
        let snap = Arc::new(registry.snapshot());
        let record = snap.providers()[0].clone();

        // Valid response → Ok(message).
        let outcome = run_provider_request(&record, r#"{"model":"m"}"#, "m");
        match outcome {
            ProviderOutcome::Ok(m) => {
                assert_eq!(m.provider, "pluggy");
                assert_eq!(m.model, "m");
                assert_eq!(m.stop_reason, rpi_ai::types::StopReason::Stop);
                assert_eq!(m.content.len(), 1);
            }
            _ => panic!("expected Ok"),
        }

        // A request fn that returns nonzero rc ⇒ Err (registered separately).
        extern "C" fn err_request_fn(
            _req: StbStringRef,
            _out: *mut StbString,
            _ud: *mut std::ffi::c_void,
        ) -> i32 {
            7
        }
        let err_record = RegisteredProvider {
            provider_id: "pluggy".to_string(),
            base_url: "https://example".to_string(),
            api_style: "anthropic-messages".to_string(),
            request_fn: err_request_fn,
            protected_request_fn: None,
            protected_stream_fn: None,
            stream_fn: None,
            plugin_free_string: stub_free,
            user_data: std::ptr::null_mut(),
        };
        let outcome = run_provider_request(&err_record, r#"{"model":"m"}"#, "m");
        assert!(matches!(outcome, ProviderOutcome::Err(_)));
    }

    /// A streaming provider's deltas must reach the event stream **in order**, as
    /// `TextStart` → `TextDelta`×N → `TextEnd`, all before the terminal `Done`.
    /// This is the property the whole feature exists for: the consumer renders
    /// while the plugin is still producing.
    #[test]
    fn stream_chunks_map_to_ordered_events() {
        let mut mapper = StreamMapper::new(rpi_ai::types::Api::Faux, "pluggy".into(), "m".into());
        let chunks = [
            serde_json::json!({"type":"start"}),
            serde_json::json!({"type":"text_start","contentIndex":0}),
            serde_json::json!({"type":"text_delta","contentIndex":0,"delta":"hel"}),
            serde_json::json!({"type":"text_delta","contentIndex":0,"delta":"lo"}),
            serde_json::json!({"type":"text_end","contentIndex":0}),
        ];
        let mut tags = Vec::new();
        for chunk in chunks {
            if let Some(event) = mapper.apply(chunk) {
                tags.push(event.type_tag());
            }
            while let Some(deferred) = mapper.take_deferred() {
                if let Some(event) = mapper.apply(deferred) {
                    tags.push(event.type_tag());
                }
            }
        }
        assert_eq!(
            tags,
            vec![
                "start",
                "text_start",
                "text_delta",
                "text_delta",
                "text_end"
            ],
            "chunks must map to an ordered, non-terminal event sequence"
        );
        assert_eq!(
            mapper.accumulated().content[0],
            rpi_ai::types::Content::text("hello"),
            "deltas accumulate into the block they name"
        );
    }

    /// The aborted terminal must map onto `ErrorReason::Aborted`, not `Error`:
    /// a user-cancelled run is not a provider fault, and the agent loop and TUI
    /// report the two differently.
    #[test]
    fn an_aborted_message_terminates_with_aborted_not_error() {
        let (mut prod, mut stream) = create_assistant_message_event_stream();
        let record = RegisteredProvider {
            provider_id: "pluggy".to_string(),
            base_url: "https://example".to_string(),
            api_style: "faux".to_string(),
            request_fn: ok_request_fn,
            protected_request_fn: None,
            protected_stream_fn: None,
            stream_fn: None,
            plugin_free_string: stub_free,
            user_data: std::ptr::null_mut(),
        };
        push_terminal(&mut prod, cancelled_message(&record));
        let event = futures::executor::block_on(stream.next()).expect("a terminal event");
        match event {
            AssistantMessageEvent::Error { reason, error } => {
                assert_eq!(reason, ErrorReason::Aborted);
                assert_eq!(error.stop_reason, rpi_ai::types::StopReason::Aborted);
            }
            other => panic!("expected Aborted Error, got {other:?}"),
        }
    }

    /// A cancelled v2 request terminates on `Aborted` promptly, and the plugin's
    /// control surface reports it — the end-to-end property the cancel protocol
    /// exists for, driven in-process rather than through the cdylib.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_v2_request_terminates_aborted_and_tells_the_plugin() {
        cancel_provider::EMITTED.store(0, Ordering::SeqCst);
        cancel_provider::OBSERVED.store(0, Ordering::SeqCst);

        let record = RegisteredProvider {
            provider_id: "cancel-probe".to_string(),
            base_url: "https://example".to_string(),
            api_style: "faux".to_string(),
            request_fn: ok_request_fn,
            protected_request_fn: None,
            protected_stream_fn: None,
            stream_fn: Some(cancel_provider::request_v2),
            plugin_free_string: stub_free,
            user_data: std::ptr::null_mut(),
        };
        let mut registry = ExtensionRegistry::new();
        registry.register_streaming_provider(record.clone());
        let snapshot = Arc::new(registry.snapshot());

        let model = Model::new(
            "m",
            "M",
            rpi_ai::types::Api::Faux,
            "cancel-probe",
            "https://example",
        );
        let ctx = Context::new(Vec::new());
        let signal = tokio_util::sync::CancellationToken::new();
        let opts = SimpleStreamOptions {
            signal: signal.clone(),
            ..SimpleStreamOptions::default()
        };

        let (mut prod, mut stream) = create_assistant_message_event_stream();
        // The driver returns the terminal message; the derived event lands on the
        // producer, so read the stream for the events the consumer would see.
        tokio::spawn(async move {
            let stream_fn = record.stream_fn.expect("a streaming fn");
            let message = drive_streaming_plugin_provider(
                &snapshot,
                &record,
                StreamProviderCallback::Legacy(stream_fn),
                Handle::current(),
                ProviderRequestLifetime::new(PluginKeepalive::empty()),
                &model,
                &ctx,
                &opts,
                None,
                &mut prod,
            )
            .await;
            push_terminal(&mut prod, message);
        });

        let mut aborted = false;
        let mut saw_delta = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while let Some(event) = stream.next().await {
            match &event {
                AssistantMessageEvent::TextDelta { .. } if !saw_delta => {
                    saw_delta = true;
                    signal.cancel();
                }
                AssistantMessageEvent::Error { reason, .. } => {
                    aborted = *reason == ErrorReason::Aborted;
                }
                _ => {}
            }
            assert!(
                std::time::Instant::now() < deadline,
                "a cancelled request must terminate promptly"
            );
        }

        assert!(saw_delta, "the provider must have started producing");
        assert!(aborted, "the terminal must be Aborted");
        // The host terminal no longer waits for the plugin to unwind.
        tokio::time::timeout(Duration::from_secs(2), async {
            while cancel_provider::OBSERVED.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the plugin observes cancellation eventually");
        assert_eq!(
            cancel_provider::OBSERVED.load(Ordering::SeqCst),
            1,
            "the plugin must have observed the cancellation through its control surface"
        );
        assert!(
            cancel_provider::EMITTED.load(Ordering::SeqCst) < 50,
            "the plugin must stop before its 50-delta budget"
        );
    }

    /// A plugin that ignores the control surface still terminates correctly: the
    /// host stops forwarding content chunks once cancelled, so `emit` reports
    /// rejection and the run ends on `Aborted` rather than hanging.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_request_stops_forwarding_content_even_if_the_plugin_pushes() {
        let owner = Arc::new(ProviderControlOwner {
            state: Arc::new(ProviderControlState::default()),
            secret: None,
            headers: Vec::new(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamChunk>();
        let bridge = StreamEmitBridge {
            tx,
            plugin_free_string: stub_free,
            control: Some(Arc::clone(&owner.state)),
        };
        let relay = &bridge as *const StreamEmitBridge as *mut c_void;

        // Not yet cancelled: a content chunk is forwarded.
        let delta = r#"{"type":"text_delta","contentIndex":0,"delta":"a"}"#;
        assert_eq!(
            stream_emit_trampoline(StbString::from_string(delta.to_string()), relay),
            0
        );
        assert!(rx.try_recv().is_ok(), "an un-cancelled chunk is delivered");

        owner.state.cancel();

        // Cancelled: the same chunk is rejected, and a structural one still lands
        // so the message can be closed.
        assert_eq!(
            stream_emit_trampoline(StbString::from_string(delta.to_string()), relay),
            1,
            "a content chunk is refused once the run is cancelled"
        );
        let end = r#"{"type":"text_end","contentIndex":0,"content":"a"}"#;
        assert_eq!(
            stream_emit_trampoline(StbString::from_string(end.to_string()), relay),
            0,
            "a structural chunk stays admissible while winding down"
        );
        assert!(
            rx.try_recv().is_err(),
            "cleanup must not produce post-terminal events"
        );
    }

    /// The control surface a v2 plugin polls: `is_cancelled` reports the latched
    /// flag for content chunks but never for structural ones (a cancelled run
    /// still needs the frames that close the message it already showed), and
    /// `wait_cancelled_ms` returns promptly once the flag is set.
    #[test]
    fn the_control_surface_reports_cancellation_for_content_only() {
        let owner = Arc::new(ProviderControlOwner {
            state: Arc::new(ProviderControlState::default()),
            secret: None,
            headers: Vec::new(),
        });
        let user_data = Arc::as_ptr(&owner) as *mut c_void;
        let content = rpi_plugin_sdk::ProviderChunkKind::Content as u32;
        let structural = rpi_plugin_sdk::ProviderChunkKind::Structural as u32;

        assert_eq!(control_is_cancelled(user_data, content), 0);
        assert_eq!(control_wait_cancelled_ms(user_data, 0), 0);

        owner.state.cancel();

        assert_eq!(
            control_is_cancelled(user_data, content),
            1,
            "content chunks must report cancelled"
        );
        assert_eq!(
            control_is_cancelled(user_data, structural),
            0,
            "structural chunks stay admissible while a cancelled request winds down"
        );
        // A blocking wait returns as soon as the (already-set) flag is seen.
        let started = std::time::Instant::now();
        assert_eq!(control_wait_cancelled_ms(user_data, 60_000), 1);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "an already-cancelled wait must not sleep out its timeout"
        );
        // A null `user_data` is the defensive floor: never report cancelled, so a
        // plugin keeps its normal behavior instead of aborting.
        assert_eq!(control_is_cancelled(std::ptr::null_mut(), content), 0);
        assert_eq!(control_wait_cancelled_ms(std::ptr::null_mut(), 0), 0);
    }

    /// A plugin that emits a delta with no preceding `start` still gets a
    /// well-formed stream: the host inserts `Start` first and re-applies the
    /// delta, so the frame encoder's `before start` rule is never violated.
    #[test]
    fn a_delta_before_start_still_opens_the_stream() {
        let mut mapper = StreamMapper::new(rpi_ai::types::Api::Faux, "pluggy".into(), "m".into());
        let first = mapper
            .apply(serde_json::json!({"type":"text_delta","contentIndex":0,"delta":"x"}))
            .expect("a start event");
        assert_eq!(first.type_tag(), "start");
        let deferred = mapper.take_deferred().expect("the parked delta");
        let second = mapper.apply(deferred).expect("the delta event");
        assert_eq!(second.type_tag(), "text_delta");
    }

    /// Tool-call deltas accumulate raw JSON and the end event carries the parsed
    /// object — the same `partial_json` invariant the Anthropic mapper holds, so
    /// a `ToolCallEnd` never contains unparseable arguments.
    #[test]
    fn tool_call_deltas_accumulate_json() {
        let mut mapper = StreamMapper::new(rpi_ai::types::Api::Faux, "pluggy".into(), "m".into());
        let mut tags = Vec::new();
        for chunk in [
            serde_json::json!({"type":"toolcall_start","contentIndex":0,"id":"c1","name":"echo"}),
            serde_json::json!({"type":"toolcall_delta","contentIndex":0,"delta":"{\"text\":"}),
            serde_json::json!({"type":"toolcall_delta","contentIndex":0,"delta":"\"hi\"}"}),
        ] {
            if let Some(event) = mapper.apply(chunk) {
                tags.push(event.type_tag());
            }
            while let Some(deferred) = mapper.take_deferred() {
                if let Some(event) = mapper.apply(deferred) {
                    tags.push(event.type_tag());
                }
            }
        }
        // The first `toolcall_start` arrives before the stream is open, so the
        // host's `start` goes out first and the chunk is re-applied right after.
        assert_eq!(
            tags,
            vec![
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta"
            ]
        );
        let event = mapper
            .apply(
                serde_json::json!({"type":"toolcall_end","contentIndex":0,"id":"c1","name":"echo"}),
            )
            .expect("tool call end");
        let AssistantMessageEvent::ToolCallEnd { tool_call, .. } = event else {
            panic!("expected ToolCallEnd");
        };
        assert_eq!(tool_call.name, "echo");
        assert_eq!(tool_call.arguments["text"], "hi");
    }
}
