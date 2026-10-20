//! Exact provider/model selection and thinking capability resolution.

use rpi_ai::{Model, ThinkingLevel};

/// Identifies a provider model and requested thinking level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    /// Exact provider identity from the host catalog.
    pub provider_id: String,
    /// Provider model identity, matched case-insensitively.
    pub model_id: String,
    /// Requested thinking level before model-specific clamping.
    pub thinking: ThinkingLevel,
}

/// Describes a catalog match and its upstream-supported thinking levels.
///
/// The result contains only catalog position and public identity metadata. It
/// does not copy model headers or other request credentials out of the host's
/// catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    /// Index of the selected model in the host-supplied catalog.
    pub catalog_index: usize,
    /// Exact provider identity from the selected model.
    pub provider_id: String,
    /// Provider model identifier from the selected model.
    pub model_id: String,
    /// Requested level before model-specific clamping.
    pub requested_thinking: ThinkingLevel,
    /// Requested level clamped to the selected model's supported levels.
    pub effective_thinking: ThinkingLevel,
    /// Ordered thinking levels reported by the selected upstream model.
    pub supported_thinking: Vec<ThinkingLevel>,
}

/// Reports a missing or ambiguous provider/model identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SelectionError {
    /// No catalog entry matched the requested provider and model.
    #[error("no model matches provider '{provider_id}' and model '{model_id}'")]
    NoMatch {
        /// Requested provider identity.
        provider_id: String,
        /// Requested model identity.
        model_id: String,
    },
    /// More than one catalog entry matched the requested identity.
    #[error("multiple models match provider '{provider_id}' and model '{model_id}'")]
    Ambiguous {
        /// Requested provider identity.
        provider_id: String,
        /// Requested model identity.
        model_id: String,
    },
}

/// Resolves a provider/model selection against a host-supplied catalog.
///
/// Provider ids are exact identities. Model ids are exact matches without
/// regard to ASCII case. The result exposes upstream thinking support and
/// clamping while keeping request headers and credentials inside the input
/// catalog.
///
/// # Parameters
/// - `catalog`: Host-supplied upstream model catalog.
/// - `selection`: Provider, model, and requested thinking identity.
///
/// # Returns
/// The matching catalog position, safe public identity, and thinking result.
///
/// # Errors
/// Returns [`SelectionError::NoMatch`] when the identity is absent or
/// [`SelectionError::Ambiguous`] when duplicate entries share that identity.
pub fn resolve_selection(
    catalog: &[Model],
    selection: &ModelSelection,
) -> Result<ResolvedSelection, SelectionError> {
    let matches = catalog
        .iter()
        .enumerate()
        .filter(|(_, model)| {
            model.provider == selection.provider_id
                && model.id.eq_ignore_ascii_case(&selection.model_id)
        })
        .collect::<Vec<_>>();

    let (catalog_index, model) = match matches.as_slice() {
        [] => {
            return Err(SelectionError::NoMatch {
                provider_id: selection.provider_id.clone(),
                model_id: selection.model_id.clone(),
            });
        }
        [matched] => *matched,
        _ => {
            return Err(SelectionError::Ambiguous {
                provider_id: selection.provider_id.clone(),
                model_id: selection.model_id.clone(),
            });
        }
    };

    let supported_thinking = model.supported_thinking_levels();
    let effective_thinking = model.clamp_thinking_level(selection.thinking);
    Ok(ResolvedSelection {
        catalog_index,
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        requested_thinking: selection.thinking,
        effective_thinking,
        supported_thinking,
    })
}
