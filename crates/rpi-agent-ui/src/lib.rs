//! Reusable model-aware product controls built on upstream `rpi-tui`.

use std::collections::HashSet;
use std::sync::Arc;

use rpi_ai::types::ThinkingLevel;
use rpi_ai::Model;
use rpi_tui::{SearchableSelectList, SelectItem, SelectList};

/// Build a searchable model selector from upstream model descriptors.
///
/// Duplicate provider/API/model entries are collapsed. The selection value is
/// provider-qualified so gateways with repeated model ids remain distinct.
///
/// # Parameters
/// - `catalog`: Models supplied by the upstream provider runtime.
/// - `current_model_id`: Model id to mark as the active choice.
///
/// # Returns
/// A searchable upstream TUI list. Register selection and cancellation
/// callbacks through [`SearchableSelectList::list`].
pub fn model_selector(catalog: &[Model], current_model_id: &str) -> Arc<SearchableSelectList> {
    let items = model_selector_items(catalog, current_model_id);
    let list = Arc::new(SelectList::new(items, 10));
    Arc::new(SearchableSelectList::new(
        Some("Select model"),
        Some("Type to filter by name, provider, or id"),
        list,
    ))
}

/// One provider row for a provider selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOption {
    /// Stable provider id returned when the row is selected.
    pub id: String,
    /// User-facing provider name.
    pub label: String,
    /// Optional secondary information shown below the label.
    pub description: Option<String>,
    /// Optional extra terms used by fuzzy filtering.
    pub search_text: Option<String>,
}

impl ProviderOption {
    /// Create a provider option.
    ///
    /// # Parameters
    /// - `id`: Stable provider id.
    /// - `label`: User-facing provider name.
    ///
    /// # Returns
    /// A provider option without optional display metadata.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: None,
            search_text: None,
        }
    }

    /// Attach secondary provider information.
    ///
    /// # Parameters
    /// - `description`: Display text shown below the provider name.
    ///
    /// # Returns
    /// This option with its display description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Attach extra fuzzy-search terms.
    ///
    /// # Parameters
    /// - `search_text`: Additional searchable provider metadata.
    ///
    /// # Returns
    /// This option with its search terms.
    pub fn with_search_text(mut self, search_text: impl Into<String>) -> Self {
        self.search_text = Some(search_text.into());
        self
    }
}

impl From<SelectItem> for ProviderOption {
    /// Convert an upstream select row to a provider option.
    ///
    /// # Parameters
    /// - `item`: Upstream select-list row.
    ///
    /// # Returns
    /// A provider option preserving display and search metadata.
    fn from(item: SelectItem) -> Self {
        Self {
            id: item.value,
            label: item.label,
            description: item.description,
            search_text: item.search_text,
        }
    }
}

/// Build a searchable provider selector from provider metadata.
///
/// # Parameters
/// - `providers`: Options supplied by the upstream provider runtime.
/// - `current_provider`: Optional provider id to mark as active.
/// - `title`: Selector heading.
/// - `description`: Search and selection hint.
///
/// # Returns
/// A searchable upstream TUI list. Register callbacks through its list.
pub fn provider_selector(
    providers: &[ProviderOption],
    current_provider: Option<&str>,
    title: &str,
    description: &str,
) -> Arc<SearchableSelectList> {
    let items = providers
        .iter()
        .map(|provider| {
            let marker = if current_provider
                .is_some_and(|current| provider.id.eq_ignore_ascii_case(current))
            {
                " (current)"
            } else {
                ""
            };
            let mut item = SelectItem::new(&provider.id, &provider.label);
            if let Some(description) = &provider.description {
                item = item.with_description(&format!("{description}{marker}"));
            } else if !marker.is_empty() {
                item = item.with_description(marker.trim());
            }
            if let Some(search_text) = &provider.search_text {
                item = item.with_search_text(search_text);
            }
            item
        })
        .collect();
    let list = Arc::new(SelectList::new(items, 10));
    Arc::new(SearchableSelectList::new(
        Some(title),
        Some(description),
        list,
    ))
}

/// Build display rows for an upstream model catalog.
///
/// # Parameters
/// - `catalog`: Models supplied by the upstream provider runtime.
/// - `current_model_id`: Model id to mark as the active choice.
///
/// # Returns
/// Provider-qualified rows suitable for an upstream `SelectList`.
pub fn model_selector_items(catalog: &[Model], current_model_id: &str) -> Vec<SelectItem> {
    let mut seen = HashSet::new();
    catalog
        .iter()
        .filter(|model| {
            seen.insert((
                model.api.clone(),
                model.provider.to_ascii_lowercase(),
                model.id.to_ascii_lowercase(),
            ))
        })
        .map(|model| {
            let label = if model.name.is_empty() {
                short_model_name(&model.id)
            } else {
                model.name.clone()
            };
            let identity = format!("{}/{}", model.provider, model.id);
            let marker = if model.id.eq_ignore_ascii_case(current_model_id) {
                " (current)"
            } else {
                ""
            };
            let name = if model.name.is_empty() {
                String::new()
            } else {
                format!(" {}", model.name)
            };
            let search_text = format!(
                "{} {}/{} {} {}{}",
                model.provider, model.provider, model.id, model.provider, model.id, name
            );
            SelectItem::new(&identity, &label)
                .with_description(&format!("{identity}{marker}"))
                .with_search_text(&search_text)
        })
        .collect()
}

/// Build a thinking-level selector from upstream-supported levels.
///
/// # Parameters
/// - `levels`: Levels returned by the selected upstream model.
///
/// # Returns
/// An upstream `SelectList` with the upstream labels and descriptions.
pub fn thinking_selector(levels: &[ThinkingLevel]) -> Arc<SelectList> {
    let items = levels
        .iter()
        .map(|level| {
            let name = thinking_level_name(*level);
            SelectItem::new(name, name).with_description(thinking_level_description(*level))
        })
        .collect();
    Arc::new(SelectList::new(items, 10))
}

/// Return the display label for an upstream thinking level.
///
/// # Parameters
/// - `level`: Upstream thinking level.
///
/// # Returns
/// The canonical lowercase label.
pub fn thinking_level_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// Return the user-facing description for an upstream thinking level.
///
/// # Parameters
/// - `level`: Upstream thinking level.
///
/// # Returns
/// A short explanation of the selected level.
pub fn thinking_level_description(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "Off — No reasoning",
        ThinkingLevel::Minimal => "Minimal — Brief reasoning (~1k tokens)",
        ThinkingLevel::Low => "Low — Light reasoning (~1k tokens)",
        ThinkingLevel::Medium => "Medium — Moderate reasoning (~80% of max)",
        ThinkingLevel::High => "High — Extensive reasoning (~95% of max)",
        ThinkingLevel::Xhigh => "Xhigh — Near-maximal reasoning",
        ThinkingLevel::Max => "Max — Maximum reasoning",
    }
}

/// Parse an upstream thinking-level label.
///
/// # Parameters
/// - `name`: Case-insensitive level name.
///
/// # Returns
/// The matching upstream level, or `None` for an unknown name.
pub fn thinking_level_from_name(name: &str) -> Option<ThinkingLevel> {
    match name.to_ascii_lowercase().as_str() {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::Xhigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

/// Shorten a provider model id for a display label.
///
/// # Parameters
/// - `id`: Provider model id.
///
/// # Returns
/// The final path/namespace segment when present.
fn short_model_name(id: &str) -> String {
    id.rsplit([':', '/'])
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(id)
        .to_string()
}
