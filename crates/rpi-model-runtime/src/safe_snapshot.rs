//! Produces versioned public model metadata without request secrets.

use rpi_ai::model::{
    AnthropicMessagesCompat, BedrockCompat, OpenaiCompletionsCompat, OpenaiResponsesCompat,
};
use rpi_ai::{InputModality, Model, ModelCost, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::ModelExecutionExclusion;

const SAFE_MODEL_SNAPSHOT_SCHEMA: &str = "rpi-safe-model-snapshot-v1";

/// Provides a versioned, secret-free display projection of an upstream model.
///
/// This projection is suitable for public model selectors and display. It is
/// not a complete execution configuration or a request-equivalence proof.
/// Inspect `excluded_execution_fields` before using it for any execution claim.
///
/// # Public members
/// - `schema_version`: Snapshot schema identifier.
/// - `provider_id`: Upstream provider identifier.
/// - `model_id`: Provider-scoped model identifier.
/// - `display_name`: Human-readable model name.
/// - `api_id`: Upstream provider API identifier.
/// - `reasoning`: Whether the model is marked as reasoning-capable.
/// - `input`: Supported input modalities.
/// - `cost`: Typed model pricing metadata.
/// - `cost_configured`: Whether pricing was explicitly configured.
/// - `context_window`: Maximum context size.
/// - `max_tokens`: Maximum output token count.
/// - `thinking_levels`: Levels accepted by upstream model selection.
/// - `compatibility`: Known typed compatibility flags, when present.
/// - `excluded_execution_fields`: Request-shaping fields omitted from output.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafeModelSnapshot {
    /// Version of this public model projection schema.
    pub schema_version: String,
    /// Public provider identifier from the upstream catalog.
    pub provider_id: String,
    /// Provider-scoped model identifier from the upstream catalog.
    pub model_id: String,
    /// Public model display name.
    pub display_name: String,
    /// Stable public provider API identifier.
    pub api_id: String,
    /// Whether the model is marked as reasoning-capable.
    pub reasoning: bool,
    /// Input modalities supported by the model.
    pub input: Vec<InputModality>,
    /// Typed upstream model pricing metadata.
    pub cost: ModelCost,
    /// Whether pricing was explicitly configured, including zero cost.
    pub cost_configured: bool,
    /// Maximum context size in tokens.
    pub context_window: u64,
    /// Maximum output token count.
    pub max_tokens: u64,
    /// Thinking levels reported by upstream model-selection logic.
    pub thinking_levels: Vec<ThinkingLevel>,
    /// Known typed compatibility flags without opaque custom values.
    pub compatibility: Option<SafeModelCompatibility>,
    /// Execution fields omitted from this display-only projection.
    pub excluded_execution_fields: Vec<ModelExecutionExclusion>,
}

/// Contains one known typed compatibility configuration.
///
/// Unknown compatibility values are intentionally not representable here.
///
/// # Public members
/// Each variant contains the corresponding typed upstream compatibility flags.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SafeModelCompatibility {
    /// Anthropic Messages API compatibility flags.
    AnthropicMessages(AnthropicMessagesCompat),
    /// OpenAI Completions API compatibility flags.
    OpenaiCompletions(OpenaiCompletionsCompat),
    /// OpenAI Responses API compatibility flags.
    OpenaiResponses(OpenaiResponsesCompat),
    /// Bedrock compatibility flags.
    Bedrock(BedrockCompat),
}

/// Captures recognized numeric request overrides from model sampling config.
///
/// Values are copied only when the property name and JSON value type match
/// this allowlist. This type does not validate provider-specific ranges or
/// claim that a provider applies every field.
///
/// # Public members
/// - `temperature`: Numeric temperature override.
/// - `top_p`: Numeric nucleus-sampling override.
/// - `frequency_penalty`: Numeric frequency penalty.
/// - `presence_penalty`: Numeric presence penalty.
/// - `seed`: Integer random seed.
/// - `max_tokens`: Integer token limit override.
/// - `max_completion_tokens`: Integer completion-token limit override.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafeSamplingParameters {
    /// Numeric temperature override.
    pub temperature: Option<f64>,
    /// Numeric nucleus-sampling override.
    pub top_p: Option<f64>,
    /// Numeric frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Numeric presence penalty.
    pub presence_penalty: Option<f64>,
    /// Integer random seed.
    pub seed: Option<i64>,
    /// Integer token limit override.
    pub max_tokens: Option<u64>,
    /// Integer completion-token limit override.
    pub max_completion_tokens: Option<u64>,
}

/// Reports safe model-level request configuration and its known omissions.
///
/// This is a partial, versioned provenance record. It includes only typed
/// numeric sampling overrides from a fixed allowlist and always reports
/// incomplete execution coverage because endpoint configuration is omitted.
/// It does not establish which provider request fields were ultimately sent.
///
/// # Public members
/// - `schema_version`: Provenance schema identifier.
/// - `provider_id`: Upstream provider identifier.
/// - `model_id`: Provider-scoped model identifier.
/// - `api_id`: Upstream provider API identifier.
/// - `sampling_parameters`: Recognized typed numeric overrides.
/// - `provider_options`: Schema-validated safe scalar provider properties.
/// - `excluded_execution_fields`: Categories of omitted request configuration.
/// - `excluded_sampling_parameter_count`: Number of unrecognized or malformed
///   sampling entries omitted without exposing their names or values.
/// - `excluded_provider_option_count`: Number of provider-specific values
///   omitted because their schema or value was not safe to preserve.
/// - `complete`: Always false while any execution field is intentionally
///   outside this safe projection.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafeModelExecutionConfiguration {
    /// Version of this partial execution provenance schema.
    pub schema_version: String,
    /// Public provider identifier from the upstream catalog.
    pub provider_id: String,
    /// Provider-scoped model identifier from the upstream catalog.
    pub model_id: String,
    /// Stable public provider API identifier.
    pub api_id: String,
    /// Recognized typed numeric request overrides.
    pub sampling_parameters: SafeSamplingParameters,
    /// Safe, schema-validated scalar provider properties.
    pub provider_options: BTreeMap<String, Value>,
    /// Execution fields omitted from this partial projection.
    pub excluded_execution_fields: Vec<ModelExecutionExclusion>,
    /// Count of omitted sampling entries without their names or values.
    pub excluded_sampling_parameter_count: usize,
    /// Count of provider properties omitted without their names or values.
    pub excluded_provider_option_count: usize,
    /// Whether all request-affecting configuration is represented.
    pub complete: bool,
}

/// Builds a safe, versioned public projection of a model.
///
/// Request endpoints, headers, credentials, arbitrary sampling values, opaque
/// provider options, provider-specific thinking labels, and opaque
/// compatibility JSON are not copied. Their presence is reported in
/// `excluded_execution_fields`. Typed display and capability metadata is
/// retained, including thinking levels as resolved by `rpi-ai`.
///
/// # Parameters
/// - `model`: Resolved upstream model descriptor.
///
/// # Returns
/// A public display snapshot that is not an execution fingerprint.
pub fn safe_model_snapshot(model: &Model) -> SafeModelSnapshot {
    let mut excluded_execution_fields = vec![ModelExecutionExclusion::Endpoint];
    if model
        .headers
        .as_ref()
        .is_some_and(|headers| !headers.is_empty())
    {
        excluded_execution_fields.push(ModelExecutionExclusion::Headers);
    }
    if !model.provider_options.is_empty() {
        excluded_execution_fields.push(ModelExecutionExclusion::ProviderOptions);
    }
    if model.sampling_params.is_some() {
        excluded_execution_fields.push(ModelExecutionExclusion::SamplingParams);
    }
    if model.thinking_level_map.is_some() {
        excluded_execution_fields.push(ModelExecutionExclusion::ThinkingLevelMap);
    }

    let compatibility = model
        .compat
        .as_ref()
        .and_then(|compatibility| match compatibility {
            rpi_ai::StreamingProtocolCompat::AnthropicMessages(value) => {
                Some(SafeModelCompatibility::AnthropicMessages(value.clone()))
            }
            rpi_ai::StreamingProtocolCompat::OpenaiCompletions(value) => {
                Some(SafeModelCompatibility::OpenaiCompletions(value.clone()))
            }
            rpi_ai::StreamingProtocolCompat::OpenaiResponses(value) => {
                Some(SafeModelCompatibility::OpenaiResponses(value.clone()))
            }
            rpi_ai::StreamingProtocolCompat::Bedrock(value) => {
                Some(SafeModelCompatibility::Bedrock(value.clone()))
            }
            rpi_ai::StreamingProtocolCompat::Other(_) => {
                excluded_execution_fields.push(ModelExecutionExclusion::OpaqueCompatibility);
                None
            }
            rpi_ai::StreamingProtocolCompat::None => None,
        });

    SafeModelSnapshot {
        schema_version: SAFE_MODEL_SNAPSHOT_SCHEMA.to_owned(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        display_name: model.name.clone(),
        api_id: model.api.as_str().to_owned(),
        reasoning: model.reasoning,
        input: model.input.clone(),
        cost: model.cost.clone(),
        cost_configured: model.cost_configured,
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        thinking_levels: model.supported_thinking_levels(),
        compatibility,
        excluded_execution_fields,
    }
}

/// Builds partial safe provenance for selected model request overrides.
///
/// The projection copies numeric values only for `temperature`, `top_p`,
/// `frequency_penalty`, `presence_penalty`, `seed`, `max_tokens`, and
/// `max_completion_tokens`. Other keys and non-numeric values are omitted and
/// counted, without returning their names or contents. Endpoint URLs, headers,
/// provider-specific thinking labels, and opaque compatibility values are
/// never included. Provider options are included only when upstream schema
/// inspection marks a scalar value safe; unsafe or unrecognized values are
/// omitted. The result is always incomplete and is not proof of the effective
/// provider request.
///
/// # Parameters
/// - `model`: Resolved upstream model descriptor.
///
/// # Returns
/// Versioned partial model execution provenance with explicit exclusions.
pub fn safe_model_execution_configuration(model: &Model) -> SafeModelExecutionConfiguration {
    let mut sampling_parameters = SafeSamplingParameters::default();
    let mut excluded_sampling_parameter_count = 0;
    if let Some(values) = &model.sampling_params {
        for (name, value) in values {
            let accepted = match name.as_str() {
                "temperature" => value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| sampling_parameters.temperature = Some(value))
                    .is_some(),
                "top_p" => value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| sampling_parameters.top_p = Some(value))
                    .is_some(),
                "frequency_penalty" => value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| sampling_parameters.frequency_penalty = Some(value))
                    .is_some(),
                "presence_penalty" => value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| sampling_parameters.presence_penalty = Some(value))
                    .is_some(),
                "seed" => value
                    .as_i64()
                    .map(|value| sampling_parameters.seed = Some(value))
                    .is_some(),
                "max_tokens" => value
                    .as_u64()
                    .map(|value| sampling_parameters.max_tokens = Some(value))
                    .is_some(),
                "max_completion_tokens" => value
                    .as_u64()
                    .map(|value| sampling_parameters.max_completion_tokens = Some(value))
                    .is_some(),
                _ => false,
            };
            if !accepted {
                excluded_sampling_parameter_count += 1;
            }
        }
    }

    let mut provider_options = BTreeMap::new();
    if let Ok(descriptors) = crate::model_property_descriptors(model) {
        for descriptor in descriptors {
            if !descriptor.is_safe_for_generic_editor() {
                continue;
            }
            let Some(value) = descriptor.value else {
                continue;
            };
            let schema_type = descriptor.schema.get("type").and_then(Value::as_str);
            let safe_scalar = match (schema_type, &value) {
                (Some("boolean"), Value::Bool(_)) => true,
                (Some("integer"), Value::Number(number)) => {
                    number.as_i64().is_some() || number.as_u64().is_some()
                }
                (Some("number"), Value::Number(number)) => {
                    number.as_f64().is_some_and(f64::is_finite)
                }
                (Some("string"), Value::String(text)) => descriptor
                    .schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| values.contains(&Value::String(text.clone()))),
                _ => false,
            };
            if safe_scalar {
                provider_options.insert(descriptor.key, value);
            }
        }
    }
    let excluded_provider_option_count = model
        .provider_options
        .len()
        .saturating_sub(provider_options.len());

    let mut excluded_execution_fields = vec![ModelExecutionExclusion::Endpoint];
    if model
        .headers
        .as_ref()
        .is_some_and(|headers| !headers.is_empty())
    {
        excluded_execution_fields.push(ModelExecutionExclusion::Headers);
    }
    if excluded_provider_option_count > 0 {
        excluded_execution_fields.push(ModelExecutionExclusion::ProviderOptions);
    }
    if excluded_sampling_parameter_count > 0 {
        excluded_execution_fields.push(ModelExecutionExclusion::SamplingParams);
    }
    if model.thinking_level_map.is_some() {
        excluded_execution_fields.push(ModelExecutionExclusion::ThinkingLevelMap);
    }
    if matches!(
        model.compat.as_ref(),
        Some(rpi_ai::StreamingProtocolCompat::Other(_))
    ) {
        excluded_execution_fields.push(ModelExecutionExclusion::OpaqueCompatibility);
    }

    SafeModelExecutionConfiguration {
        schema_version: "rpi-safe-model-execution-v1".to_owned(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        api_id: model.api.as_str().to_owned(),
        sampling_parameters,
        provider_options,
        excluded_execution_fields,
        excluded_sampling_parameter_count,
        excluded_provider_option_count,
        complete: false,
    }
}

#[cfg(test)]
#[path = "safe_snapshot_tests.rs"]
mod tests;
