//! Adapts host-authorized isolated worker providers to the upstream Provider trait.

use std::sync::{Arc, Mutex};

use super::super::protocol::{
    WorkerProviderOptions, WorkerProviderOutcome, WorkerProviderRequest, WorkerProviderSpec,
};
use super::{IsolatedWorker, WorkerError};
use rpi_ai::event_stream::create_assistant_message_event_stream;
use rpi_ai::{
    AppliedRequestSettingsCallback, AssistantMessage, AssistantMessageEvent,
    AssistantMessageEventStream, Context, DoneReason, ErrorReason, Model, Provider, RequestSecret,
    SimpleStreamOptions, StopReason,
};

/// A Provider adapter that executes extension callbacks in an isolated worker.
///
/// # Public members
/// - [`WorkerProvider::from_worker`]: Builds adapters from worker registrations.
/// - [`Provider::id`]: Returns the host-validated provider identity.
/// - [`Provider::models`]: Returns credential-free advertised model metadata.
/// - [`Provider::stream_simple`]: Executes an isolated provider request.
///
/// # Private members
/// - `id`: Provider identity returned by the worker.
/// - `models`: Sanitized model catalog returned by the worker.
/// - `worker`: Shared process and protocol connection.
pub struct WorkerProvider {
    /// Provider identity returned by the worker.
    id: String,
    /// Sanitized model catalog returned by the worker.
    models: Vec<Model>,
    /// Shared process and protocol connection.
    worker: Arc<Mutex<IsolatedWorker>>,
}

impl WorkerProvider {
    /// Creates provider adapters for all worker-advertised provider models.
    ///
    /// The worker's model metadata is treated as untrusted: host credentials in
    /// model headers are removed and each model's provider id is host-stamped.
    ///
    /// # Parameters
    /// - `worker`: Shared worker process to use for provider requests.
    ///
    /// # Returns
    /// One provider adapter for each advertised provider with at least one model.
    pub fn from_worker(worker: Arc<Mutex<IsolatedWorker>>) -> Vec<Arc<dyn Provider>> {
        let registrations = worker
            .lock()
            .map(|worker| worker.providers().to_vec())
            .unwrap_or_default();
        registrations
            .into_iter()
            .filter_map(|registration| Self::from_registration(worker.clone(), registration))
            .map(|provider| provider as Arc<dyn Provider>)
            .collect()
    }

    /// Creates one adapter after validating provider and model identities.
    ///
    /// # Parameters
    /// - `worker`: Shared worker process.
    /// - `registration`: Worker-advertised provider and models.
    ///
    /// # Returns
    /// The sanitized adapter, or `None` when the registration is unusable.
    fn from_registration(
        worker: Arc<Mutex<IsolatedWorker>>,
        registration: WorkerProviderSpec,
    ) -> Option<Arc<Self>> {
        if registration.id.trim().is_empty() {
            return None;
        }
        let mut model_ids = std::collections::BTreeSet::new();
        let models = registration
            .models
            .into_iter()
            .filter_map(|mut model| {
                if model.id.trim().is_empty() || !model_ids.insert(model.id.clone()) {
                    return None;
                }
                model.provider.clone_from(&registration.id);
                model.headers = None;
                Some(model)
            })
            .collect::<Vec<_>>();
        (!models.is_empty()).then_some(Arc::new(Self {
            id: registration.id,
            models,
            worker,
        }))
    }

    /// Starts one protected provider request and returns its event stream.
    ///
    /// # Parameters
    /// - `model`: Advertised provider model selected by the host.
    /// - `context`: Provider-neutral context for this turn.
    /// - `options`: Non-secret upstream request options.
    /// - `secret`: Optional host-protected request credential and bearer mode.
    ///
    /// # Returns
    /// An event stream containing the isolated provider's terminal response.
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        secret: Option<(&RequestSecret, bool)>,
    ) -> AssistantMessageEventStream {
        let (mut producer, stream) = create_assistant_message_event_stream();
        let partial = Arc::new(AssistantMessage::empty(
            model.api.clone(),
            self.id.clone(),
            model.id.clone(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis() as i64),
        ));
        producer.push(AssistantMessageEvent::Start { partial });
        let request_secret = secret
            .map(|(secret, _)| secret_to_string(secret))
            .or_else(|| options.api_key.clone());
        let bearer = secret.is_some_and(|(_, bearer)| bearer);
        let request = WorkerProviderRequest {
            provider: self.id.clone(),
            model: model.id.clone(),
            context: context.clone(),
            options: WorkerProviderOptions {
                timeout_ms: options.timeout.map(|timeout| timeout.as_millis() as u64),
                max_retries: options.max_retries,
                reasoning: options.reasoning,
                max_tokens: options.max_tokens,
                temperature: options.temperature,
            },
            secret: request_secret,
            bearer,
        };
        let worker = Arc::clone(&self.worker);
        let cancellation = options.signal.clone();
        let request_cancellation = cancellation.clone();
        let provider_id = self.id.clone();
        let model = model.clone();
        tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                let worker = worker.lock().map_err(|_| {
                    WorkerError::Closed("worker process lock is poisoned".to_owned())
                })?;
                worker.call_provider_cancellable(request, Some(&request_cancellation))
            })
            .await;
            let response = match result {
                Ok(Ok(response)) => response,
                _ => {
                    let error = AssistantMessage::terminal(
                        model.api.clone(),
                        provider_id,
                        model.id.clone(),
                        if cancellation.is_cancelled() {
                            StopReason::Aborted
                        } else {
                            StopReason::Error
                        },
                        "isolated provider request failed",
                        0,
                    );
                    let reason = if cancellation.is_cancelled() {
                        ErrorReason::Aborted
                    } else {
                        ErrorReason::Error
                    };
                    producer.push(AssistantMessageEvent::Error { reason, error });
                    return;
                }
            };
            let mut message = response.message;
            message.provider = provider_id;
            message.api = model.api.clone();
            message.model.clone_from(&model.id);
            match response.outcome {
                WorkerProviderOutcome::Stop => producer.push(AssistantMessageEvent::Done {
                    reason: DoneReason::Stop,
                    message,
                }),
                WorkerProviderOutcome::Length => producer.push(AssistantMessageEvent::Done {
                    reason: DoneReason::Length,
                    message,
                }),
                WorkerProviderOutcome::ToolUse => producer.push(AssistantMessageEvent::Done {
                    reason: DoneReason::ToolUse,
                    message,
                }),
                WorkerProviderOutcome::Deferred => producer.push(AssistantMessageEvent::Done {
                    reason: DoneReason::Deferred,
                    message,
                }),
                WorkerProviderOutcome::Authentication => {
                    producer.push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Authentication,
                        error: message,
                    })
                }
                WorkerProviderOutcome::Error => producer.push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Error,
                    error: message,
                }),
                WorkerProviderOutcome::Aborted => producer.push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Aborted,
                    error: message,
                }),
            };
        });
        stream
    }
}

#[async_trait::async_trait]
impl Provider for WorkerProvider {
    /// Returns the host-stamped provider id.
    fn id(&self) -> &str {
        &self.id
    }

    /// Returns the credential-free worker model catalog.
    fn models(&self) -> &[Model] {
        &self.models
    }

    /// Executes a one-shot provider request in the worker process.
    async fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options, None)
    }

    /// Executes a provider request with a protected host credential.
    async fn stream_simple_with_request_secret(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options, Some((secret, bearer)))
    }

    /// Executes a provider request and reports that extension request settings
    /// remain opaque to the host.
    async fn stream_simple_with_applied_settings(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        callback(incomplete_settings(model));
        self.stream(model, context, options, None)
    }

    /// Executes a protected provider request and reports opaque settings.
    async fn stream_simple_with_request_secret_and_applied_settings(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        callback(incomplete_settings(model));
        self.stream(model, context, options, Some((secret, bearer)))
    }
}

/// Copies opaque credential bytes only for the provider call frame.
///
/// # Parameters
/// - `secret`: Host-protected provider request credential.
///
/// # Returns
/// The secret value to send on the worker's private stdin pipe.
fn secret_to_string(secret: &RequestSecret) -> String {
    let mut value = String::new();
    secret.with_secret(&mut |secret| value.push_str(secret));
    value
}

/// Builds an honest incomplete report for an opaque extension provider.
///
/// # Parameters
/// - `model`: Exact model selected by the host.
///
/// # Returns
/// A report that does not claim visibility into extension request transforms.
fn incomplete_settings(model: &Model) -> rpi_ai::AppliedRequestSettings {
    rpi_ai::AppliedRequestSettings {
        schema_version: "rpi-applied-request-settings-v1".to_owned(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        api_id: model.api.as_str().to_owned(),
        applied_values: Default::default(),
        request_body_sha256: None,
        request_header_sha256: None,
        omitted_categories: vec![
            "extension_managed_request_transforms".to_owned(),
            "provider_applied_values_unavailable".to_owned(),
            "provider_options_values".to_owned(),
            "endpoint".to_owned(),
            "headers_and_credentials".to_owned(),
            "prompt_and_transcript".to_owned(),
        ],
        complete: false,
    }
}
