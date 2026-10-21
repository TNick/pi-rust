//! Mirrors `packages/ai/src/models.ts` — `Model<TApi>` + the compatibility structs.
//!
//! The compat structs (`AnthropicMessagesCompat`, etc.) are kept here rather than in
//! `types.rs` because they are model-layer configuration, not the core message
//! contract. Providers read them off `Model` to decide request shaping.

use crate::error::AiError;
use crate::types::{Api, InputModality, ModelCost, ProviderId, ThinkingLevelMap};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Versioned schema for provider-specific model properties.
///
/// This schema version is independent of the provider request-envelope
/// version. Values are validated and coerced by the provider runtime before
/// request shaping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProviderOptionsSchema {
    /// Version of the JSON Schema envelope.
    pub version: u32,
    /// Object schema describing supported provider-specific properties.
    pub schema: Value,
}

/// Compatibility flags for Anthropic Messages-compatible APIs.
/// Mirrors TS `AnthropicMessagesCompat`. `Default` matches the documented defaults
/// at each field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
    /// Default: true.
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Default: true.
    pub supports_long_cache_retention: Option<bool>,
    /// Default: false.
    pub send_session_affinity_headers: Option<bool>,
    /// Default: true.
    pub supports_cache_control_on_tools: Option<bool>,
    /// Default: true.
    pub supports_temperature: Option<bool>,
    /// Default: false.
    pub force_adaptive_thinking: Option<bool>,
    /// Default: false.
    pub allow_empty_signature: Option<bool>,
    /// Default: false.
    pub supports_strict_tools: Option<bool>,
    /// Default: true for first-party Anthropic (except Haiku / Claude ≤ 4.4); false otherwise.
    pub supports_tool_references: Option<bool>,
}

impl AnthropicMessagesCompat {
    /// Resolve a flag to its effective value given the documented default.
    pub fn eager_tool_input_streaming(&self) -> bool {
        self.supports_eager_tool_input_streaming.unwrap_or(true)
    }
    pub fn long_cache_retention(&self) -> bool {
        self.supports_long_cache_retention.unwrap_or(true)
    }
    pub fn cache_control_on_tools(&self) -> bool {
        self.supports_cache_control_on_tools.unwrap_or(true)
    }
    pub fn temperature(&self) -> bool {
        self.supports_temperature.unwrap_or(true)
    }
    pub fn adaptive_thinking(&self) -> bool {
        self.force_adaptive_thinking.unwrap_or(false)
    }
    pub fn empty_signature(&self) -> bool {
        self.allow_empty_signature.unwrap_or(false)
    }
    pub fn strict_tools(&self) -> bool {
        self.supports_strict_tools.unwrap_or(false)
    }
    pub fn tool_references(&self) -> bool {
        // Mirrors defaultSupportsToolReferences: true for first-party Anthropic
        // except Haiku and Claude ≤ 4.4. The model catalog overrides this for
        // non-Anthropic providers (false).
        self.supports_tool_references.unwrap_or(true)
    }
}

/// Whether the next request carries the model's own prior reasoning back to it.
///
/// This decides what the system prompt may claim about the working-state
/// channel: when reasoning *is* replayed, telling the model "reasoning is not
/// carried into your next turn" is simply false, and the instruction to restate
/// the plan every turn becomes busy-work (the re-plan symptom in
/// `docs/debugging/llm-repetition-forensics.md`).
///
/// - `openai-completions`: with `compat.requiresThinkingAsText` (text parts) or
///   `compat.requiresReasoningContent` (a separate reasoning_content field).
/// - `openai-responses`: visible thinking is replayed as input content.
/// - `anthropic-messages`: thinking blocks are echoed back with their signature.
/// - everything else: reasoning is dropped, so the model cannot see it again.
pub fn prior_reasoning_is_replayed(model: &Model) -> bool {
    match model.api {
        Api::OpenaiCompletions => matches!(
            model.compat.as_ref(),
            Some(StreamingProtocolCompat::OpenaiCompletions(compat))
                if compat.requires_thinking_as_text == Some(true)
                    || compat.requires_reasoning_content == Some(true)
        ),
        Api::OpenaiResponses | Api::AzureOpenaiResponses | Api::OpenaiCodexResponses => true,
        Api::AnthropicMessages => true,
        _ => false,
    }
}

/// Compatibility flags for OpenAI Completions-compat APIs. Only a subset is
/// modeled for v1 (we have no OpenAI provider yet); the rest live here as
/// `Option`s for forward-compat.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenaiCompletionsCompat {
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_usage_in_streaming: Option<bool>,
    pub supports_finish_reason: Option<bool>,
    pub max_tokens_field: Option<MaxTokensField>,
    pub requires_tool_result_name: Option<bool>,
    pub requires_assistant_after_tool_result: Option<bool>,
    pub requires_thinking_as_text: Option<bool>,
    /// Replay prior assistant thinking in the wire reasoning_content field.
    /// Opt-in per model; ordinary OpenAI endpoints keep their existing shape.
    pub requires_reasoning_content: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    pub send_session_affinity_headers: Option<bool>,
    pub supports_long_cache_retention: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    MaxCompletionTokens,
    MaxTokens,
}

/// Compatibility flags for OpenAI Responses-compat APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenaiResponsesCompat {
    pub supports_developer_role: Option<bool>,
    pub supports_long_cache_retention: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    /// Whether this endpoint accepts OpenAI custom tools with Lark/regex
    /// grammar formats. Grammar tools fall back to ordinary function tools
    /// when this is unset or false.
    #[serde(rename = "supportsOpenAIGrammarTools")]
    pub supports_openai_grammar_tools: Option<bool>,
}

/// Compatibility flags for Bedrock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct BedrockCompat {
    pub supports_strict_mode: Option<bool>,
}

/// Per-API compat blob. Matches the TS `compat?` discriminator on `Model`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StreamingProtocolCompat {
    AnthropicMessages(AnthropicMessagesCompat),
    OpenaiCompletions(OpenaiCompletionsCompat),
    OpenaiResponses(OpenaiResponsesCompat),
    Bedrock(BedrockCompat),
    /// Custom / unknown API compat (opaque passthrough).
    Other(serde_json::Value),
    /// No compat configured (default for faux and simple providers).
    None,
}

impl Default for StreamingProtocolCompat {
    fn default() -> Self {
        StreamingProtocolCompat::None
    }
}

impl StreamingProtocolCompat {
    pub fn as_anthropic(&self) -> Option<&AnthropicMessagesCompat> {
        match self {
            StreamingProtocolCompat::AnthropicMessages(c) => Some(c),
            _ => None,
        }
    }
}

/// A model entry. Mirrors TS `Model<TApi>`; the generic is erased to a runtime
/// `Api` field so a single `Model` works for every provider.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<InputModality>,
    pub cost: ModelCost,
    /// Whether pricing was explicitly supplied, including intentionally free models.
    #[serde(default)]
    pub cost_configured: bool,
    pub context_window: u64,
    pub max_tokens: u64,
    /// Values merged into OpenAI-compatible request bodies when present.
    /// Other built-in adapters may ignore this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<StreamingProtocolCompat>,
    /// Runtime-only provider-specific values validated by `provider_options_schema`.
    ///
    /// Values are excluded from ordinary serialization and redacted in Debug.
    #[serde(skip)]
    pub provider_options: BTreeMap<String, serde_json::Value>,
    /// Versioned schema for `provider_options`; schema values are safe metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options_schema: Option<ProviderOptionsSchema>,
}

impl std::fmt::Debug for Model {
    /// Formats model metadata while redacting request credentials and options.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("reasoning", &self.reasoning)
            .field("thinking_level_map", &self.thinking_level_map)
            .field("input", &self.input)
            .field("cost", &self.cost)
            .field("cost_configured", &self.cost_configured)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .field("sampling_params", &self.sampling_params)
            .field("headers", &self.headers.as_ref().map(|_| "[redacted]"))
            .field("compat", &self.compat)
            .field(
                "provider_options",
                &(!self.provider_options.is_empty()).then_some("[redacted]"),
            )
            .field("provider_options_schema", &self.provider_options_schema)
            .finish()
    }
}

impl Model {
    /// Construct a minimal model with empty cost + zeroed windows.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        api: Api,
        provider: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api,
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            cost_configured: false,
            context_window: 0,
            max_tokens: 0,
            sampling_params: None,
            headers: None,
            compat: None,
            provider_options: BTreeMap::new(),
            provider_options_schema: None,
        }
    }

    /// Validate and coerce configured provider-specific model properties.
    ///
    /// Property schemas must be finite object schemas with
    /// `additionalProperties: false`. Request-envelope and credential fields
    /// cannot be declared as provider-specific properties.
    ///
    /// # Returns
    /// The normalized property map safe to apply to this model's request.
    ///
    /// # Errors
    /// Returns an error when values are present without a schema or fail it.
    pub fn validated_provider_options(&self) -> Result<Map<String, Value>, AiError> {
        let Some(properties) = self.provider_option_schema_properties()? else {
            if self.provider_options.is_empty() {
                return Ok(Map::new());
            }
            return Err(AiError::Schema {
                message: "provider options require a versioned schema".to_owned(),
            });
        };
        let schema = self
            .provider_options_schema
            .as_ref()
            .expect("checked schema");
        let values = Value::Object(
            self.provider_options
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let validated = crate::schema::validate_json_value(values, &schema.schema)?;
        let options = validated
            .as_object()
            .cloned()
            .ok_or_else(|| AiError::Schema {
                message: "provider options must be an object".to_owned(),
            })?;
        if options.keys().any(|key| !properties.contains_key(key)) {
            return Err(AiError::Schema {
                message: "provider options contain an undeclared property".to_owned(),
            });
        }
        Ok(options)
    }

    /// Return the editable top-level property schemas after checking the schema
    /// version and the request/credential reserved-name boundary.
    pub fn provider_option_schema_properties(
        &self,
    ) -> Result<Option<&Map<String, Value>>, AiError> {
        let Some(schema) = self.provider_options_schema.as_ref() else {
            return Ok(None);
        };
        if schema.version != 1 {
            return Err(AiError::Schema {
                message: "unsupported provider options schema version".to_owned(),
            });
        }
        let schema = self
            .provider_options_schema
            .as_ref()
            .expect("schema checked above");
        let Some(schema_object) = schema.schema.as_object() else {
            return Err(AiError::Schema {
                message: "provider options schema must be an object schema".to_owned(),
            });
        };
        if schema_object.get("type").and_then(Value::as_str) != Some("object")
            || schema_object.get("additionalProperties") != Some(&Value::Bool(false))
        {
            return Err(AiError::Schema {
                message: "provider options schema must declare a closed object".to_owned(),
            });
        }
        let properties = schema_object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| AiError::Schema {
                message: "provider options schema must declare properties".to_owned(),
            })?;
        if properties
            .keys()
            .any(|key| !provider_option_key_allowed(key, &self.api))
        {
            return Err(AiError::Schema {
                message: "provider options schema contains a reserved property".to_owned(),
            });
        }
        Ok(Some(properties))
    }

    /// Explicit zero rates and built-in nonzero rates both count as known pricing.
    pub fn has_pricing(&self) -> bool {
        self.cost_configured || self.cost != ModelCost::default()
    }

    /// Returns the supported thinking levels for this model. Mirrors
    /// `getSupportedThinkingLevels`: `xhigh`/`max` are only listed when their map
    /// entry is a non-null string; other levels are supported unless mapped to `null`.
    pub fn supported_thinking_levels(&self) -> Vec<crate::types::ThinkingLevel> {
        use crate::types::ThinkingLevel::*;
        let map = self.thinking_level_map.as_ref();
        let supported = |lvl: crate::types::ThinkingLevel| -> bool {
            match map.and_then(|m| m.get(&lvl)) {
                Some(None) => false,   // explicitly null → unsupported
                Some(Some(_)) => true, // mapped → supported
                None => true,          // absent → use provider default (supported)
            }
        };
        let mut levels = vec![Off, Minimal, Low, Medium, High];
        // xhigh/max only when their map entry is a non-null string.
        if matches!(map.and_then(|m| m.get(&Xhigh)), Some(Some(_))) {
            levels.push(Xhigh);
        }
        if matches!(map.and_then(|m| m.get(&Max)), Some(Some(_))) {
            levels.push(Max);
        }
        levels.retain(|&lvl| supported(lvl));
        levels
    }

    /// Clamp a requested thinking level to a supported one. Mirrors
    /// `clampThinkingLevel`: walks the ordered list and accepts if supported,
    /// otherwise falls back to the nearest lower supported level.
    pub fn clamp_thinking_level(
        &self,
        requested: crate::types::ThinkingLevel,
    ) -> crate::types::ThinkingLevel {
        use crate::types::ThinkingLevel::*;
        let order = [Off, Minimal, Low, Medium, High, Xhigh, Max];
        let supported: std::collections::HashSet<_> =
            self.supported_thinking_levels().into_iter().collect();
        let req_idx = order
            .iter()
            .position(|l| *l == requested)
            .unwrap_or(order.len() - 1);
        // Walk from requested down to find nearest supported.
        for i in (0..=req_idx).rev() {
            if supported.contains(&order[i]) {
                return order[i];
            }
        }
        Off
    }
}

/// Rejects request-envelope and credential properties in a model schema.
///
/// # Parameters
/// - `key`: Provider-specific property name.
/// - `api`: Provider protocol receiving the property.
///
/// # Returns
/// `true` when the property can be merged into that protocol's request body.
fn provider_option_key_allowed(key: &str, api: &Api) -> bool {
    let normalized = key.to_ascii_lowercase();
    let secret_fields = [
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "credentials",
        "headers",
        "password",
        "secret",
        "token",
    ];
    if secret_fields.contains(&normalized.as_str()) {
        return false;
    }
    let shared_request_fields = [
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "response_format",
        "reasoning_effort",
        "thinking",
        "thinking_budget",
        "reasoning_budget",
        "stream_options",
    ];
    if shared_request_fields.contains(&normalized.as_str()) {
        return false;
    }
    let reserved = match api {
        Api::OpenaiCompletions => &[
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "reasoning_effort",
            "tools",
            "store",
        ][..],
        Api::OpenaiResponses => &[
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "max_output_tokens",
            "reasoning",
            "tools",
        ][..],
        Api::AnthropicMessages => &[
            "model",
            "messages",
            "max_tokens",
            "stream",
            "system",
            "thinking",
            "tools",
            "output_config",
        ][..],
        _ => &["model", "stream", "tools"][..],
    };
    !reserved.contains(&normalized.as_str())
}
