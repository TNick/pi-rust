//! Generic property editor built on the upstream settings-list component.
//!
//! Hosts adapt their own validated property descriptors into these rows and
//! handle changes through callbacks. This component does not own a schema or
//! validation rules.

use std::any::Any;
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent};

use crate::component::Component;
use crate::settings_list::{SettingChoice, SettingItem, SettingsList};
use crate::terminal::InputEvent;

/// A schema-neutral setting row for a host-provided property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyEditorItem {
    /// Stable host property key.
    pub key: String,
    /// Human-readable label.
    pub label: String,
    /// Current formatted value.
    pub value: String,
    /// Optional help text.
    pub description: Option<String>,
    /// Allowed values for a finite selector; empty values open an edit action.
    pub choices: Vec<PropertyEditorChoice>,
    /// Opaque value matching one finite choice, when applicable.
    pub selected_choice: Option<String>,
}

/// One finite choice with a display label and opaque host value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyEditorChoice {
    /// Text displayed in the selector.
    pub label: String,
    /// Opaque value sent back to the host callback.
    pub value: String,
}

/// Reusable settings-list surface for editing host-provided properties.
pub struct PropertyEditor {
    list: SettingsList,
}

impl PropertyEditor {
    /// Creates an editor from host-adapted rows.
    pub fn new(items: Vec<PropertyEditorItem>) -> Self {
        let rows = items.into_iter().map(to_setting_item).collect();
        Self {
            list: SettingsList::new(rows),
        }
    }

    /// Installs the callback used for values selected from finite choices.
    pub fn on_change(&self, callback: Arc<dyn Fn(&str, &str) + Send + Sync>) {
        self.list.on_change(callback);
    }

    /// Installs the callback used when a free-form row is activated.
    pub fn on_edit(&self, callback: Arc<dyn Fn(&SettingItem) + Send + Sync>) {
        self.list.on_select(callback);
    }

    /// Installs the callback used when Escape is pressed.
    pub fn on_cancel(&self, callback: Arc<dyn Fn() + Send + Sync>) {
        self.list.on_cancel(callback);
    }

    /// Returns the current formatted value for a property key.
    pub fn value_of(&self, key: &str) -> Option<String> {
        self.list.value_of(key)
    }

    /// Updates a row after a host-side edit has been validated.
    pub fn set_value(&self, key: &str, value: &str) {
        self.list.update_value(key, value);
    }

    /// Replaces all displayed rows.
    pub fn set_items(&self, items: Vec<PropertyEditorItem>) {
        self.list
            .set_items(items.into_iter().map(to_setting_item).collect());
    }

    /// Routes a key event through the shared settings-list behavior.
    pub fn handle_key(&self, key: KeyEvent) {
        self.list.handle_key(key);
    }
}

impl Component for PropertyEditor {
    fn render(&self, width: usize) -> Vec<String> {
        self.list.render(width)
    }

    fn handle_event(&self, event: &InputEvent) -> bool {
        if let InputEvent::Key(key) = event {
            if matches!(
                key.code,
                KeyCode::Up | KeyCode::Down | KeyCode::Enter | KeyCode::Esc | KeyCode::Char(' ')
            ) {
                self.list.handle_key(*key);
                return true;
            }
        }
        false
    }

    fn invalidate(&self) {
        self.list.invalidate();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Converts a host row into the shared settings-list representation.
fn to_setting_item(item: PropertyEditorItem) -> SettingItem {
    let current = item.selected_choice.as_deref().unwrap_or(&item.value);
    let mut row = SettingItem::new(&item.key, &item.label, current);
    if let Some(description) = item.description {
        row = row.with_description(&description);
    }
    if item.choices.is_empty() {
        row.with_submenu()
    } else {
        let choices = item
            .choices
            .into_iter()
            .map(|choice| SettingChoice {
                label: choice.label,
                value: choice.value,
            })
            .collect::<Vec<_>>();
        row.with_choices(&choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn finite_choices_notify_host_and_update_display() {
        let editor = PropertyEditor::new(vec![PropertyEditorItem {
            key: "mode".into(),
            label: "Mode".into(),
            value: "Fast".into(),
            description: None,
            choices: vec![
                PropertyEditorChoice { label: "Fast".into(), value: "fast".into() },
                PropertyEditorChoice { label: "Safe".into(), value: "safe".into() },
            ],
            selected_choice: Some("fast".into()),
        }]);
        let observed = Arc::new(Mutex::new(None));
        let observed_cb = observed.clone();
        editor.on_change(Arc::new(move |key, value| {
            *observed_cb.lock().unwrap() = Some((key.to_owned(), value.to_owned()));
        }));

        editor.handle_key(KeyEvent::from(KeyCode::Enter));

        assert_eq!(editor.value_of("mode").as_deref(), Some("safe"));
        assert_eq!(
            *observed.lock().unwrap(),
            Some(("mode".into(), "safe".into()))
        );
        assert!(editor.render(80).iter().any(|line| line.contains("Safe")));
    }

    #[test]
    fn typed_enum_values_keep_distinct_labels_and_payloads() {
        let editor = PropertyEditor::new(vec![PropertyEditorItem {
            key: "enabled".into(),
            label: "Enabled".into(),
            value: "String true".into(),
            description: None,
            choices: vec![
                PropertyEditorChoice {
                    label: "String true".into(),
                    value: "\"true\"".into(),
                },
                PropertyEditorChoice {
                    label: "Boolean true".into(),
                    value: "true".into(),
                },
            ],
            selected_choice: Some("\"true\"".into()),
        }]);
        let observed = Arc::new(Mutex::new(None));
        let observed_cb = observed.clone();
        editor.on_change(Arc::new(move |key, value| {
            *observed_cb.lock().unwrap() = Some((key.to_owned(), value.to_owned()));
        }));

        editor.handle_key(KeyEvent::from(KeyCode::Enter));

        assert_eq!(
            *observed.lock().unwrap(),
            Some(("enabled".into(), "true".into()))
        );
        assert!(editor.render(80).iter().any(|line| line.contains("Boolean true")));
    }

    #[test]
    fn free_form_row_uses_edit_callback() {
        let editor = PropertyEditor::new(vec![PropertyEditorItem {
            key: "endpoint".into(),
            label: "Endpoint".into(),
            value: "".into(),
            description: Some("Provider-specific endpoint".into()),
            choices: Vec::new(),
            selected_choice: None,
        }]);
        let edited = Arc::new(Mutex::new(None));
        let edited_cb = edited.clone();
        editor.on_edit(Arc::new(move |item| {
            *edited_cb.lock().unwrap() = Some(item.key.clone());
        }));

        editor.handle_key(KeyEvent::from(KeyCode::Enter));

        assert_eq!(*edited.lock().unwrap(), Some("endpoint".into()));
    }
}
