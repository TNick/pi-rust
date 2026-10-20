//! Creates reusable built-in provider implementations from resolved models.

use std::sync::Arc;

use rpi_ai::provider::DeferredProvider;
use rpi_ai::{
    Api, AppliedRequestSettingsCallback, AssistantMessageEventStream, Context, Model, Provider,
    RequestSecret, SimpleStreamOptions,
};

/// Reports invalid provider/model bindings or an unavailable built-in API.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuiltinProviderError {
    /// The resolved provider identity was empty.
    #[error("provider identity must not be empty")]
    EmptyProviderId,
    /// The host did not supply any resolved models.
    #[error("at least one resolved model is required")]
    EmptyModels,
    /// A resolved model had no model identity.
    #[error("resolved model identity must not be empty")]
    EmptyModelId,
    /// A resolved model belongs to another provider identity.
    #[error("model '{model_id}' belongs to provider '{actual}', expected '{expected}'")]
    ProviderMismatch {
        /// Expected public provider identity.
        expected: String,
        /// Actual public provider identity.
        actual: String,
        /// Model identity with the mismatch.
        model_id: String,
    },
    /// The supplied models use more than one provider protocol.
    #[error("resolved models use more than one provider API")]
    MixedApis,
    /// The selected provider API has no built-in provider implementation.
    #[error("no built-in provider is registered for API '{0}'")]
    UnsupportedApi(String),
}

/// Creates an upstream built-in provider for one resolved provider catalog.
///
/// The host supplies already-authorized models and a configured HTTP client.
/// This API creates only the protocol implementation; it does not discover
/// providers, read credentials, or make package/configuration policy choices.
///
/// # Parameters
/// - `provider_id`: Public provider identity that owns every supplied model.
/// - `models`: Resolved model descriptors for that provider.
/// - `http`: Host-configured HTTP client used by provider requests.
///
/// # Returns
/// The protocol implementation bound to the supplied provider models.
///
/// # Errors
/// Returns an error when the model binding is empty, inconsistent, or uses an
/// API without a built-in provider implementation.
pub fn create_builtin_provider(
    provider_id: &str,
    models: Vec<Model>,
    http: reqwest::Client,
) -> Result<Arc<dyn Provider>, BuiltinProviderError> {
    let Some(first) = models.first() else {
        return Err(BuiltinProviderError::EmptyModels);
    };
    create_builtin_provider_for_api(provider_id, first.api.clone(), models, http)
}

/// Creates a built-in provider for an explicit API, including an empty catalog.
///
/// This variant is useful while constructing CLI runtimes before model
/// availability has been resolved. An empty provider exposes no models and
/// rejects every request through its normal binding check.
///
/// # Parameters
/// - `provider_id`: Public provider identity for every supplied model.
/// - `api`: Protocol selected by the enclosing configuration.
/// - `models`: Resolved descriptors, possibly empty.
/// - `http`: Host-configured HTTP client.
///
/// # Returns
/// The selected protocol implementation bound to the supplied models.
///
/// # Errors
/// Returns an error for an empty provider identity, inconsistent model binding,
/// or an API without a built-in implementation.
pub fn create_builtin_provider_for_api(
    provider_id: &str,
    api: Api,
    models: Vec<Model>,
    http: reqwest::Client,
) -> Result<Arc<dyn Provider>, BuiltinProviderError> {
    if provider_id.is_empty() {
        return Err(BuiltinProviderError::EmptyProviderId);
    }

    for model in &models {
        if model.id.is_empty() {
            return Err(BuiltinProviderError::EmptyModelId);
        }
        if model.provider != provider_id {
            return Err(BuiltinProviderError::ProviderMismatch {
                expected: provider_id.to_owned(),
                actual: model.provider.clone(),
                model_id: model.id.clone(),
            });
        }
        if model.api != api {
            return Err(BuiltinProviderError::MixedApis);
        }
    }
    let bound_models = models.clone();

    let inner: Arc<dyn Provider> = match api {
        Api::AnthropicMessages => Arc::new(
            rpi_ai::providers::anthropic::AnthropicProvider::with_models_without_env_api_key(
                None, http, models,
            ),
        ),
        Api::OpenaiCompletions => Arc::new(
            rpi_ai::providers::openai_completions::OpenAiCompletionsProvider::with_models_without_env_api_key(
                provider_id, None, http, models,
            ),
        ),
        Api::OpenaiResponses => Arc::new(
            rpi_ai::providers::openai_responses::OpenAiResponsesProvider::with_models_without_env_api_key(
                provider_id, None, http, models,
            ),
        ),
        other => {
            return Err(BuiltinProviderError::UnsupportedApi(
                other.as_str().to_owned(),
            ));
        }
    };

    let descriptors = bound_models.iter().map(safe_model_descriptor).collect();

    Ok(Arc::new(BuiltinProvider {
        id: provider_id.to_owned(),
        inner,
        bound_models,
        models: descriptors,
    }))
}

/// Removes request-private values from a provider's public model descriptor.
///
/// # Parameters
/// - `model`: Canonical provider model.
///
/// # Returns
/// A descriptor without model headers or opaque provider options.
fn safe_model_descriptor(model: &Model) -> Model {
    let mut descriptor = model.clone();
    descriptor.headers = None;
    descriptor.provider_options.clear();
    descriptor
}

/// Wraps a protocol provider with the host-selected public identity.
struct BuiltinProvider {
    /// Host-selected public provider identity.
    id: String,
    /// Upstream protocol implementation receiving the requests.
    inner: Arc<dyn Provider>,
    /// Canonical models used to reject caller-substituted request settings.
    bound_models: Vec<Model>,
    /// Safe model descriptors exposed to provider consumers.
    models: Vec<Model>,
}

#[async_trait::async_trait]
impl Provider for BuiltinProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn models(&self) -> &[Model] {
        &self.models
    }

    async fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let Some(model) = self.bound_model(model) else {
            return unbound_model_error(model, &self.id);
        };
        self.inner.stream_simple(&model, context, options).await
    }

    async fn stream_simple_with_applied_settings(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        let Some(model) = self.bound_model(model) else {
            return unbound_model_error(model, &self.id);
        };
        self.inner
            .stream_simple_with_applied_settings(&model, context, options, callback)
            .await
    }

    async fn stream_simple_with_request_secret(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
    ) -> AssistantMessageEventStream {
        let Some(model) = self.bound_model(model) else {
            return unbound_model_error(model, &self.id);
        };
        self.inner
            .stream_simple_with_request_secret(&model, context, options, secret, bearer)
            .await
    }

    async fn stream_simple_with_request_secret_and_applied_settings(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
        callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        let Some(model) = self.bound_model(model) else {
            return unbound_model_error(model, &self.id);
        };
        self.inner
            .stream_simple_with_request_secret_and_applied_settings(
                &model, context, options, secret, bearer, callback,
            )
            .await
    }

    fn deferred(&self) -> Option<&dyn DeferredProvider> {
        self.inner.deferred()
    }
}

impl BuiltinProvider {
    /// Resolve a caller model to the canonical descriptor bound at construction.
    ///
    /// # Parameters
    /// - `requested`: Caller-selected model descriptor.
    ///
    /// # Returns
    /// The canonical model when provider, API, and model identity match.
    fn bound_model(&self, requested: &Model) -> Option<Model> {
        self.bound_models
            .iter()
            .find(|model| {
                requested.provider == self.id
                    && model.provider == requested.provider
                    && model.api == requested.api
                    && model.id.eq_ignore_ascii_case(&requested.id)
            })
            .cloned()
    }
}

/// Returns an error stream for a model outside the provider's bound catalog.
///
/// # Parameters
/// - `model`: Caller-supplied model rejected by the binding.
/// - `provider_id`: Provider identity receiving the attempted request.
///
/// # Returns
/// A stream with one terminal error event.
fn unbound_model_error(model: &Model, provider_id: &str) -> AssistantMessageEventStream {
    let (mut producer, stream) = rpi_ai::event_stream::create_assistant_message_event_stream();
    producer.push(rpi_ai::AssistantMessageEvent::Error {
        reason: rpi_ai::types::ErrorReason::Error,
        error: rpi_ai::types::AssistantMessage::terminal(
            model.api.clone(),
            provider_id,
            model.id.clone(),
            rpi_ai::types::StopReason::Error,
            "model is not registered in this provider execution binding",
            0,
        ),
    });
    stream
}

/// Reports whether an API has a reusable built-in provider implementation.
///
/// # Parameters
/// - `api`: Provider protocol to inspect.
///
/// # Returns
/// `true` when [`create_builtin_provider`] can create that protocol.
#[must_use]
pub fn supports_builtin_provider_api(api: &Api) -> bool {
    matches!(
        api,
        Api::AnthropicMessages | Api::OpenaiCompletions | Api::OpenaiResponses
    )
}
