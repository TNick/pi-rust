//! Hierarchical selectable list component with bounded viewport rendering.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crossterm::event::KeyCode;

use crate::component::Component;
use crate::terminal::InputEvent;
use crate::utils::truncate_to_width;

/// One caller-identified item in a tree list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeItem {
    /// Stable caller-provided item identity.
    pub id: String,
    /// Text displayed for the item.
    pub label: String,
    /// Parent identity, or `None` for a root item.
    pub parent_id: Option<String>,
}

/// One visible tree row with caller identity and computed hierarchy state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeRow {
    /// Stable identity of the represented item.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Depth from a root item, where roots have depth zero.
    pub depth: usize,
    /// Whether the item has child items.
    pub has_children: bool,
    /// Whether its child items are currently visible.
    pub expanded: bool,
    /// Whether this row is selected.
    pub selected: bool,
}

#[derive(Default)]
struct TreeState {
    selected: Option<String>,
    expanded: HashSet<String>,
    scroll_offset: usize,
}

/// A keyboard-navigable tree list with stable IDs and bounded visible rows.
pub struct TreeList {
    items: Vec<TreeItem>,
    item_index: HashMap<String, usize>,
    children: HashMap<Option<String>, Vec<String>>,
    depths: HashMap<String, usize>,
    max_visible: usize,
    state: Mutex<TreeState>,
}

impl TreeList {
    /// Creates a tree list and validates item identities and parent links.
    ///
    /// # Parameters
    /// - `items`: Items in caller-defined sibling order.
    /// - `max_visible`: Maximum number of rows returned for display.
    ///
    /// # Returns
    /// A tree list with all root items expanded.
    ///
    /// # Errors
    /// Returns an error for duplicate or empty IDs, missing parents, or cycles.
    pub fn new(items: Vec<TreeItem>, max_visible: usize) -> Result<Self, String> {
        let mut item_index = HashMap::new();
        for (index, item) in items.iter().enumerate() {
            if item.id.is_empty() || item_index.insert(item.id.clone(), index).is_some() {
                return Err(format!(
                    "tree item ID is empty or duplicated: `{}`",
                    item.id
                ));
            }
        }

        let mut children: HashMap<Option<String>, Vec<String>> = HashMap::new();
        for item in &items {
            if let Some(parent) = &item.parent_id {
                if !item_index.contains_key(parent) {
                    return Err(format!(
                        "tree item `{}` has missing parent `{parent}`",
                        item.id
                    ));
                }
            }
            children
                .entry(item.parent_id.clone())
                .or_default()
                .push(item.id.clone());
        }

        let mut depths = HashMap::new();
        for item in &items {
            let depth = item_depth(item, &item_index, &items)?;
            depths.insert(item.id.clone(), depth);
        }

        let roots = children.get(&None).cloned().unwrap_or_default();
        let mut expanded = HashSet::new();
        for root in roots {
            if children.contains_key(&Some(root.clone())) {
                expanded.insert(root);
            }
        }
        let selected = items.first().map(|item| item.id.clone());
        Ok(Self {
            items,
            item_index,
            children,
            depths,
            max_visible: max_visible.max(1),
            state: Mutex::new(TreeState {
                selected,
                expanded,
                scroll_offset: 0,
            }),
        })
    }

    /// Returns the currently selected caller-provided ID.
    pub fn selected_id(&self) -> Option<String> {
        self.state.lock().ok()?.selected.clone()
    }

    /// Selects an item by stable caller-provided ID.
    ///
    /// # Parameters
    /// - `id`: Identity of an existing item.
    ///
    /// # Returns
    /// `true` when the item exists and becomes selected.
    pub fn select(&self, id: &str) -> bool {
        if !self.item_index.contains_key(id) {
            return false;
        }
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let mut parent = self
            .item_index
            .get(id)
            .and_then(|index| self.items[*index].parent_id.clone());
        while let Some(parent_id) = parent {
            state.expanded.insert(parent_id.clone());
            parent = self
                .item_index
                .get(&parent_id)
                .and_then(|index| self.items[*index].parent_id.clone());
        }
        state.selected = Some(id.to_owned());
        self.keep_selection_visible(&mut state);
        true
    }

    /// Expands or collapses a parent item by its stable ID.
    ///
    /// Collapsing an ancestor of the selected row moves selection to that
    /// ancestor so selection always refers to a visible row.
    ///
    /// # Parameters
    /// - `id`: Identity of the parent item.
    /// - `expanded`: Desired expansion state.
    ///
    /// # Returns
    /// `true` when the item exists and has children.
    pub fn set_expanded(&self, id: &str, expanded: bool) -> bool {
        if !self.has_children(id) {
            return false;
        }
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if expanded {
            state.expanded.insert(id.to_owned());
        } else {
            state.expanded.remove(id);
            if state
                .selected
                .as_deref()
                .is_some_and(|selected| self.is_descendant(selected, id))
            {
                state.selected = Some(id.to_owned());
            }
        }
        self.keep_selection_visible(&mut state);
        true
    }

    /// Toggles expansion for the currently selected parent item.
    pub fn toggle_selected_expansion(&self) -> bool {
        let Some(id) = self.selected_id() else {
            return false;
        };
        let expanded = self
            .state
            .lock()
            .map(|state| state.expanded.contains(&id))
            .unwrap_or(false);
        self.set_expanded(&id, !expanded)
    }

    /// Returns the bounded visible rows in current display order.
    pub fn visible_rows(&self) -> Vec<TreeRow> {
        let Ok(mut state) = self.state.lock() else {
            return Vec::new();
        };
        let rows = self.flatten_visible(&state);
        if rows.is_empty() {
            state.scroll_offset = 0;
            return rows;
        }
        let selected = rows.iter().position(|row| row.selected).unwrap_or_default();
        state.scroll_offset =
            bounded_offset(state.scroll_offset, selected, rows.len(), self.max_visible);
        rows.into_iter()
            .skip(state.scroll_offset)
            .take(self.max_visible)
            .collect()
    }

    /// Moves selection by one visible row in the requested direction.
    fn move_selection(&self, delta: isize) {
        let rows =
            self.flatten_visible(&self.state.lock().unwrap_or_else(|error| error.into_inner()));
        if rows.is_empty() {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let current = rows.iter().position(|row| row.selected).unwrap_or_default();
        let next = current.saturating_add_signed(delta).min(rows.len() - 1);
        state.selected = Some(rows[next].id.clone());
        self.keep_selection_visible(&mut state);
    }

    /// Keeps the selected row inside the current bounded viewport.
    fn keep_selection_visible(&self, state: &mut TreeState) {
        let rows = self.flatten_visible(state);
        let Some(selected) = state
            .selected
            .as_deref()
            .and_then(|id| rows.iter().position(|row| row.id == id))
        else {
            return;
        };
        state.scroll_offset =
            bounded_offset(state.scroll_offset, selected, rows.len(), self.max_visible);
    }

    /// Flattens expanded branches into rows in caller-defined sibling order.
    fn flatten_visible(&self, state: &TreeState) -> Vec<TreeRow> {
        fn append(
            tree: &TreeList,
            state: &TreeState,
            parent: Option<String>,
            rows: &mut Vec<TreeRow>,
        ) {
            let Some(children) = tree.children.get(&parent) else {
                return;
            };
            for id in children {
                let Some(index) = tree.item_index.get(id) else {
                    continue;
                };
                let item = &tree.items[*index];
                let has_children = tree.children.contains_key(&Some(id.clone()));
                let expanded = state.expanded.contains(id);
                rows.push(TreeRow {
                    id: item.id.clone(),
                    label: item.label.clone(),
                    depth: tree.depths.get(id).copied().unwrap_or_default(),
                    has_children,
                    expanded,
                    selected: state.selected.as_deref() == Some(id),
                });
                if expanded {
                    append(tree, state, Some(id.clone()), rows);
                }
            }
        }

        let mut rows = Vec::new();
        append(self, state, None, &mut rows);
        rows
    }

    /// Reports whether an item has child items.
    fn has_children(&self, id: &str) -> bool {
        self.children.contains_key(&Some(id.to_owned()))
    }

    /// Checks whether one item is below another in the validated parent graph.
    fn is_descendant(&self, item_id: &str, ancestor_id: &str) -> bool {
        let mut current = self
            .item_index
            .get(item_id)
            .and_then(|index| self.items[*index].parent_id.as_deref());
        while let Some(parent) = current {
            if parent == ancestor_id {
                return true;
            }
            current = self
                .item_index
                .get(parent)
                .and_then(|index| self.items[*index].parent_id.as_deref());
        }
        false
    }
}

impl Component for TreeList {
    /// Renders only the current bounded rows.
    fn render(&self, width: usize) -> Vec<String> {
        self.visible_rows()
            .into_iter()
            .map(|row| {
                let marker = if row.has_children {
                    if row.expanded {
                        "▾ "
                    } else {
                        "▸ "
                    }
                } else {
                    "  "
                };
                let prefix = format!(
                    "{}{}{}",
                    if row.selected { "› " } else { "  " },
                    "  ".repeat(row.depth),
                    marker
                );
                let remaining = width.saturating_sub(prefix.chars().count());
                format!("{}{}", prefix, truncate_to_width(&row.label, remaining, ""))
            })
            .collect()
    }

    /// Handles arrow and activation keys for selection and branch expansion.
    fn handle_event(&self, event: &InputEvent) -> bool {
        let InputEvent::Key(key) = event else {
            return false;
        };
        match key.code {
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Left => {
                if !self.set_expanded_selected(false) {
                    return false;
                }
            }
            KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                if !self.toggle_selected_expansion() {
                    return false;
                }
            }
            _ => return false,
        }
        true
    }

    /// Invalidates no cached content because rows are rendered from current state.
    fn invalidate(&self) {}

    /// Enables component downcasting.
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TreeList {
    /// Collapses the selected parent or selects its visible parent row.
    fn set_expanded_selected(&self, expanded: bool) -> bool {
        let Some(id) = self.selected_id() else {
            return false;
        };
        if self.set_expanded(&id, expanded) {
            return true;
        }
        if expanded {
            return false;
        }
        let Some(parent) = self
            .item_index
            .get(&id)
            .and_then(|index| self.items[*index].parent_id.as_deref())
            .map(str::to_owned)
        else {
            return false;
        };
        self.select(&parent)
    }
}

/// Computes a checked depth for one item and rejects parent cycles.
fn item_depth(
    item: &TreeItem,
    item_index: &HashMap<String, usize>,
    items: &[TreeItem],
) -> Result<usize, String> {
    let mut depth = 0;
    let mut current = item.parent_id.as_deref();
    let mut visited = HashSet::new();
    visited.insert(item.id.as_str());
    while let Some(parent) = current {
        if !visited.insert(parent) {
            return Err(format!("tree item parent cycle includes `{parent}`"));
        }
        let index = item_index
            .get(parent)
            .ok_or_else(|| format!("tree item `{}` has missing parent `{parent}`", item.id))?;
        depth += 1;
        current = items[*index].parent_id.as_deref();
    }
    Ok(depth)
}

/// Keeps the selected row visible and clamps the viewport to valid bounds.
fn bounded_offset(offset: usize, selected: usize, row_count: usize, limit: usize) -> usize {
    let max_offset = row_count.saturating_sub(limit);
    let offset = if selected < offset {
        selected
    } else if selected >= offset.saturating_add(limit) {
        selected.saturating_add(1).saturating_sub(limit)
    } else {
        offset
    };
    offset.min(max_offset)
}

#[cfg(test)]
mod tests {
    //! Tests for generic tree structure, selection, and viewport behavior.

    use crate::component::Component;

    use super::{TreeItem, TreeList};

    fn item(id: &str, parent: Option<&str>) -> TreeItem {
        TreeItem {
            id: id.to_owned(),
            label: id.to_owned(),
            parent_id: parent.map(str::to_owned),
        }
    }

    /// Preserves caller IDs and parent-derived depth in visible rows.
    #[test]
    fn renders_expanded_hierarchy_in_caller_order() {
        let tree = TreeList::new(
            vec![
                item("root", None),
                item("child", Some("root")),
                item("leaf", Some("child")),
            ],
            8,
        )
        .unwrap();
        assert_eq!(
            tree.visible_rows()
                .iter()
                .map(|row| (row.id.as_str(), row.depth))
                .collect::<Vec<_>>(),
            vec![("root", 0), ("child", 1)]
        );
        assert!(tree.set_expanded("child", true));
        assert_eq!(tree.visible_rows()[2].id, "leaf");
    }

    /// Collapse hides descendants and moves selection to the collapsed parent.
    #[test]
    fn collapse_keeps_selection_on_a_visible_row() {
        let tree = TreeList::new(vec![item("root", None), item("child", Some("root"))], 8).unwrap();
        tree.select("child");
        assert!(tree.set_expanded("root", false));
        assert_eq!(tree.selected_id().as_deref(), Some("root"));
        assert_eq!(tree.visible_rows().len(), 1);
    }

    /// Scrolls the selected row into a bounded viewport.
    #[test]
    fn viewport_scrolls_and_render_never_exceeds_its_row_limit() {
        let tree =
            TreeList::new(vec![item("a", None), item("b", None), item("c", None)], 2).unwrap();
        assert!(tree.select("c"));
        assert_eq!(
            tree.visible_rows()
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        assert_eq!(tree.render(20).len(), 2);
    }

    /// Rejects duplicate IDs, unresolved parents, and cycles.
    #[test]
    fn rejects_invalid_tree_identity_and_structure() {
        assert!(TreeList::new(vec![item("x", None), item("x", None)], 4).is_err());
        assert!(TreeList::new(vec![item("x", Some("missing"))], 4).is_err());
        assert!(TreeList::new(vec![item("a", Some("b")), item("b", Some("a"))], 4).is_err());
    }
}
