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
    visible_rows: Vec<TreeRow>,
    visible_index: HashMap<String, usize>,
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
        let (item_index, children, depths) = build_indexes(&items)?;
        let roots = children.get(&None).cloned().unwrap_or_default();
        let mut expanded = HashSet::new();
        for root in roots {
            if children.contains_key(&Some(root.clone())) {
                expanded.insert(root);
            }
        }
        let selected = items.first().map(|item| item.id.clone());
        let tree = Self {
            items,
            item_index,
            children,
            depths,
            max_visible: max_visible.max(1),
            state: Mutex::new(TreeState {
                selected,
                expanded,
                scroll_offset: 0,
                visible_rows: Vec::new(),
                visible_index: HashMap::new(),
            }),
        };
        if let Ok(mut state) = tree.state.lock() {
            tree.rebuild_visible(&mut state);
        }
        Ok(tree)
    }

    /// Returns the current caller-provided items in display order.
    pub fn items(&self) -> &[TreeItem] {
        &self.items
    }

    /// Replaces every item while preserving stable selection and expansion state.
    ///
    /// The complete replacement is validated before any tree state changes.
    /// Selection is retained when the selected ID still exists; otherwise the
    /// first new item becomes selected, or selection is cleared when no items
    /// remain. Expansion is retained only for surviving parents that still have
    /// children, and newly added parents are expanded so newly projected content
    /// is visible. Removed IDs are dropped from selection and expansion state.
    ///
    /// # Parameters
    /// - `items`: Complete replacement in caller-defined sibling order.
    ///
    /// # Returns
    /// `Ok(())` when indexes and cached visible rows were rebuilt.
    ///
    /// # Errors
    /// Returns an error for duplicate or empty IDs, missing parents, cycles, or
    /// a poisoned tree-state lock. Invalid input leaves the previous items and
    /// selection state unchanged.
    pub fn replace_items(&mut self, items: Vec<TreeItem>) -> Result<(), String> {
        let (item_index, children, depths) = build_indexes(&items)?;
        let Ok(mut state) = self.state.lock() else {
            return Err("tree list state lock is poisoned".to_owned());
        };
        let previous_ids = self.item_index.keys().cloned().collect::<HashSet<_>>();
        self.items = items;
        self.item_index = item_index;
        self.children = children;
        self.depths = depths;
        state.expanded.retain(|id| self.has_children(id));
        for item in &self.items {
            if !previous_ids.contains(&item.id) && self.has_children(&item.id) {
                state.expanded.insert(item.id.clone());
            }
        }
        if state
            .selected
            .as_ref()
            .is_some_and(|id| !self.item_index.contains_key(id))
        {
            state.selected = self.items.first().map(|item| item.id.clone());
        }
        if let Some(selected) = state.selected.clone() {
            self.expand_ancestors(&mut state, &selected);
        }
        self.rebuild_visible(&mut state);
        self.keep_selection_visible(&mut state);
        Ok(())
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
        let expanded = self.expand_ancestors(&mut state, id);
        if expanded {
            state.selected = Some(id.to_owned());
            self.rebuild_visible(&mut state);
        } else {
            self.set_selected(&mut state, id.to_owned());
        }
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
        let changed = if expanded {
            state.expanded.insert(id.to_owned())
        } else {
            let changed = state.expanded.remove(id);
            if state
                .selected
                .as_deref()
                .is_some_and(|selected| self.is_descendant(selected, id))
            {
                state.selected = Some(id.to_owned());
            }
            changed
        };
        if changed {
            self.rebuild_visible(&mut state);
        }
        self.keep_selection_visible(&mut state);
        true
    }

    /// Changes expansion state for several parent IDs with one visible-row rebuild.
    ///
    /// This is useful when a caller initializes or reveals a large subtree. It
    /// avoids rebuilding the complete visible preorder once for each parent.
    /// Unknown IDs and leaf IDs are ignored.
    ///
    /// # Parameters
    /// - `ids`: Parent identities whose expansion state should change.
    /// - `expanded`: Whether each parent should expose its children.
    ///
    /// # Returns
    /// The number of parent expansion states that changed.
    pub fn set_expanded_many(&self, ids: &[String], expanded: bool) -> usize {
        let Ok(mut state) = self.state.lock() else {
            return 0;
        };
        let mut changed = 0;
        for id in ids {
            if !self.has_children(id) {
                continue;
            }
            let did_change = if expanded {
                state.expanded.insert(id.clone())
            } else {
                state.expanded.remove(id)
            };
            changed += usize::from(did_change);
        }
        if changed > 0 {
            self.rebuild_visible(&mut state);
            self.keep_selection_visible(&mut state);
        }
        changed
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
        if state.visible_rows.is_empty() {
            state.scroll_offset = 0;
            return Vec::new();
        }
        let selected = state
            .selected
            .as_deref()
            .and_then(|id| state.visible_index.get(id).copied())
            .unwrap_or_default();
        state.scroll_offset = bounded_offset(
            state.scroll_offset,
            selected,
            state.visible_rows.len(),
            self.max_visible,
        );
        state
            .visible_rows
            .iter()
            .skip(state.scroll_offset)
            .take(self.max_visible)
            .cloned()
            .collect()
    }

    /// Moves selection by one visible row in the requested direction.
    fn move_selection(&self, delta: isize) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.visible_rows.is_empty() {
            return;
        }
        let current = state
            .selected
            .as_deref()
            .and_then(|id| state.visible_index.get(id).copied())
            .unwrap_or_default();
        let next = current
            .saturating_add_signed(delta)
            .min(state.visible_rows.len() - 1);
        let id = state.visible_rows[next].id.clone();
        self.set_selected(&mut state, id);
        self.keep_selection_visible(&mut state);
    }

    /// Keeps the selected row inside the current bounded viewport.
    fn keep_selection_visible(&self, state: &mut TreeState) {
        let Some(selected) = state
            .selected
            .as_deref()
            .and_then(|id| state.visible_index.get(id).copied())
        else {
            return;
        };
        state.scroll_offset = bounded_offset(
            state.scroll_offset,
            selected,
            state.visible_rows.len(),
            self.max_visible,
        );
    }

    /// Rebuilds the expanded preorder cache after tree structure changes.
    fn rebuild_visible(&self, state: &mut TreeState) {
        let mut rows = Vec::new();
        let mut pending = self
            .children
            .get(&None)
            .into_iter()
            .flatten()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            let Some(index) = self.item_index.get(&id).copied() else {
                continue;
            };
            let item = &self.items[index];
            let has_children = self.has_children(&id);
            let expanded = state.expanded.contains(&id);
            rows.push(TreeRow {
                id: item.id.clone(),
                label: item.label.clone(),
                depth: self.depths.get(&id).copied().unwrap_or_default(),
                has_children,
                expanded,
                selected: state.selected.as_deref() == Some(&id),
            });
            if expanded {
                if let Some(children) = self.children.get(&Some(id)) {
                    pending.extend(children.iter().rev().cloned());
                }
            }
        }
        state.visible_index = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.id.clone(), index))
            .collect();
        state.visible_rows = rows;
    }

    /// Updates the selected identity and markers in constant time.
    fn set_selected(&self, state: &mut TreeState, id: String) {
        if let Some(previous) = state.selected.replace(id.clone()) {
            if let Some(index) = state.visible_index.get(&previous) {
                if let Some(row) = state.visible_rows.get_mut(*index) {
                    row.selected = false;
                }
            }
        }
        if let Some(index) = state.visible_index.get(&id) {
            if let Some(row) = state.visible_rows.get_mut(*index) {
                row.selected = true;
            }
        }
    }

    /// Marks every ancestor of one item expanded so the item becomes visible.
    ///
    /// # Parameters
    /// - `state`: Locked tree state receiving expansion markers.
    /// - `id`: Existing item whose ancestor path should be visible.
    ///
    /// # Returns
    /// `true` when at least one ancestor was newly expanded.
    fn expand_ancestors(&self, state: &mut TreeState, id: &str) -> bool {
        let mut expanded = false;
        let mut parent = self
            .item_index
            .get(id)
            .and_then(|index| self.items[*index].parent_id.clone());
        while let Some(parent_id) = parent {
            expanded |= state.expanded.insert(parent_id.clone());
            parent = self
                .item_index
                .get(&parent_id)
                .and_then(|index| self.items[*index].parent_id.clone());
        }
        expanded
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

/// Validates a complete item set and builds its lookup indexes.
///
/// # Parameters
/// - `items`: Items in caller-defined sibling order.
///
/// # Returns
/// Item-position index, ordered children by parent, and checked item depths.
///
/// # Errors
/// Returns an error for duplicate or empty IDs, missing parents, or cycles.
fn build_indexes(
    items: &[TreeItem],
) -> Result<
    (
        HashMap<String, usize>,
        HashMap<Option<String>, Vec<String>>,
        HashMap<String, usize>,
    ),
    String,
> {
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
    for item in items {
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

    let depths = item_depths(items, &item_index)?;
    Ok((item_index, children, depths))
}

/// Computes all checked item depths in linear time and rejects parent cycles.
fn item_depths(
    items: &[TreeItem],
    item_index: &HashMap<String, usize>,
) -> Result<HashMap<String, usize>, String> {
    let mut depths = HashMap::with_capacity(items.len());
    for item in items {
        if depths.contains_key(&item.id) {
            continue;
        }
        let mut path = Vec::new();
        let mut visiting = HashSet::new();
        let mut current = item;
        let mut depth: usize = loop {
            if let Some(depth) = depths.get(&current.id) {
                break *depth;
            }
            if !visiting.insert(current.id.as_str()) {
                return Err(format!("tree item parent cycle includes `{}`", current.id));
            }
            path.push(current);
            let Some(parent) = current.parent_id.as_deref() else {
                break 0;
            };
            let index = item_index.get(parent).ok_or_else(|| {
                format!("tree item `{}` has missing parent `{parent}`", current.id)
            })?;
            current = &items[*index];
        };
        for item in path.into_iter().rev() {
            if item.parent_id.is_some() {
                depth = depth
                    .checked_add(1)
                    .ok_or_else(|| "tree nesting exceeds supported depth".to_owned())?;
            } else {
                depth = 0;
            }
            depths.insert(item.id.clone(), depth);
        }
    }
    Ok(depths)
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

    /// Applies a batch of parent expansion changes and retains valid selection.
    #[test]
    fn batch_expansion_updates_all_parents_before_rebuilding_visible_rows() {
        let tree = TreeList::new(
            vec![
                item("root", None),
                item("first", Some("root")),
                item("leaf", Some("first")),
                item("second", Some("root")),
                item("other-leaf", Some("second")),
            ],
            8,
        )
        .unwrap();
        let parents = vec![
            "first".to_owned(),
            "second".to_owned(),
            "missing".to_owned(),
        ];

        assert_eq!(tree.set_expanded_many(&parents, true), 2);
        assert_eq!(
            tree.visible_rows()
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "first", "leaf", "second", "other-leaf"]
        );
        assert_eq!(tree.set_expanded_many(&parents, false), 2);
        assert_eq!(tree.visible_rows().len(), 3);
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

    /// Renders a deep tree without recursively traversing its expanded path.
    #[test]
    fn deep_tree_uses_cached_visible_rows_and_bounded_rendering() {
        let size = 4096;
        let items = (0..size)
            .map(|index| {
                let id = format!("node-{index}");
                let parent_id = (index > 0).then(|| format!("node-{}", index - 1));
                TreeItem {
                    id,
                    label: format!("Node {index}"),
                    parent_id,
                }
            })
            .collect();
        let tree = TreeList::new(items, 16).unwrap();
        let parents = (0..size - 1)
            .map(|index| format!("node-{index}"))
            .collect::<Vec<_>>();
        assert_eq!(tree.set_expanded_many(&parents, true), size - 2);

        assert!(tree.select(&format!("node-{}", size - 1)));

        let rows = tree.visible_rows();
        assert_eq!(rows.len(), 16);
        assert_eq!(
            rows.last().map(|row| row.id.as_str()),
            Some(format!("node-{}", size - 1).as_str())
        );
        assert_eq!(rows.last().map(|row| row.depth), Some(size - 1));
        assert!(rows.last().is_some_and(|row| row.selected));
    }

    /// Rejects duplicate IDs, unresolved parents, and cycles.
    #[test]
    fn rejects_invalid_tree_identity_and_structure() {
        assert!(TreeList::new(vec![item("x", None), item("x", None)], 4).is_err());
        assert!(TreeList::new(vec![item("x", Some("missing"))], 4).is_err());
        assert!(TreeList::new(vec![item("a", Some("b")), item("b", Some("a"))], 4).is_err());
    }

    /// Retains selection and expansion for IDs that survive replacement.
    #[test]
    fn replacement_preserves_stable_selection_and_expansion() {
        let mut tree = TreeList::new(
            vec![
                item("root", None),
                item("a", Some("root")),
                item("b", Some("root")),
                item("other", None),
            ],
            8,
        )
        .unwrap();
        assert!(tree.select("a"));
        assert!(tree
            .replace_items(vec![
                item("root", None),
                item("a", Some("root")),
                item("c", Some("root")),
                item("other", None),
            ])
            .is_ok());

        assert_eq!(tree.selected_id().as_deref(), Some("a"));
        let rows = tree.visible_rows();
        assert!(rows.iter().any(|row| row.id == "c"));
        assert!(!rows.iter().any(|row| row.id == "b"));
    }

    /// Leaves collapsed surviving parents collapsed and expands new parents.
    #[test]
    fn replacement_keeps_collapsed_parents_and_expands_new_parents() {
        let mut tree = TreeList::new(
            vec![
                item("root", None),
                item("parent", Some("root")),
                item("leaf", Some("parent")),
                item("other", None),
            ],
            8,
        )
        .unwrap();
        assert!(tree.select("leaf"));
        assert!(tree.select("other"));
        assert!(tree.set_expanded("parent", false));
        assert!(tree
            .replace_items(vec![
                item("root", None),
                item("parent", Some("root")),
                item("leaf", Some("parent")),
                item("other", None),
                item("new-parent", Some("root")),
                item("new-leaf", Some("new-parent")),
            ])
            .is_ok());

        assert_eq!(tree.selected_id().as_deref(), Some("other"));
        let rows = tree.visible_rows();
        assert!(!rows.iter().any(|row| row.id == "leaf"));
        assert!(rows.iter().any(|row| row.id == "new-leaf"));
    }

    /// Falls back predictably when the selected ID disappears.
    #[test]
    fn replacement_selects_the_first_surviving_item_when_selection_is_removed() {
        let mut tree = TreeList::new(vec![item("a", None), item("b", None)], 8).unwrap();
        assert!(tree.select("b"));
        assert!(tree.replace_items(vec![item("a", None)]).is_ok());

        assert_eq!(tree.selected_id().as_deref(), Some("a"));
        assert_eq!(tree.items().len(), 1);
        assert_eq!(tree.visible_rows().len(), 1);
    }

    /// Rejects invalid replacements without changing current items or selection.
    #[test]
    fn invalid_replacement_preserves_the_previous_tree() {
        let mut tree = TreeList::new(vec![item("a", None)], 8).unwrap();
        assert!(tree
            .replace_items(vec![item("a", None), item("a", None)])
            .is_err());

        assert_eq!(tree.items().len(), 1);
        assert_eq!(tree.selected_id().as_deref(), Some("a"));
    }
}
