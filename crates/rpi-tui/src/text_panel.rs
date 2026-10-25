//! Renders passive, text-only panels for reusable terminal hosts.

use std::sync::{Mutex, PoisonError};

use crate::{strip_ansi, theme, truncate_to_width, visible_width, Component, ThemeColors};

const MAX_PANEL_TEXT_CHARS: usize = 4_096;
const MAX_HIGHLIGHT_CHARS: usize = 256;
const MAX_HIGHLIGHTS: usize = 32;

/// Selects a theme color for one literal text highlight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextPanelHighlightColor {
    /// Uses the active theme's warning color.
    Warning,
    /// Uses the active theme's error color.
    Error,
}

/// Describes literal text to emphasize inside a panel line.
///
/// # Public members
/// - `text`: Literal substring to highlight.
/// - `color`: Theme color applied to matching text.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPanelHighlight {
    /// Literal substring to highlight.
    pub text: String,
    /// Theme color applied to matching text.
    pub color: TextPanelHighlightColor,
}

/// Provides safe content and geometry for a passive text panel.
///
/// The host controls panel placement. This value controls only bounded text
/// layout and rendering, so it can be adapted from any extension protocol.
///
/// # Public members
/// - `width`: Maximum panel width in terminal columns.
/// - `max_height`: Maximum rendered row count.
/// - `border`: Whether to draw a dim border.
/// - `title`: Optional title rendered before content lines.
/// - `lines`: Plain text content lines.
/// - `highlights`: Literal substrings colored by the active theme.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPanelOptions {
    /// Maximum panel width in terminal columns.
    pub width: usize,
    /// Maximum rendered row count.
    pub max_height: usize,
    /// Whether to draw a dim border.
    pub border: bool,
    /// Optional title rendered before content lines.
    pub title: String,
    /// Plain text content lines.
    pub lines: Vec<String>,
    /// Literal substrings colored by the active theme.
    pub highlights: Vec<TextPanelHighlight>,
}

/// Renders mutable passive text-panel content as a reusable TUI component.
///
/// # Public members
/// - [`TextPanel::new`]: Creates a panel with initial content.
/// - [`TextPanel::set_options`]: Replaces the panel content and layout.
///
/// # Private members
/// - `options`: Current bounded panel description.
pub struct TextPanel {
    /// Current bounded panel description.
    options: Mutex<TextPanelOptions>,
}

impl TextPanel {
    /// Creates a panel component with the supplied content and layout.
    ///
    /// # Parameters
    /// - `options`: Panel width, height, text, border, and highlights.
    ///
    /// # Returns
    /// A reusable component that renders the supplied panel.
    pub fn new(options: TextPanelOptions) -> Self {
        Self {
            options: Mutex::new(options),
        }
    }

    /// Replaces the content and layout used by subsequent renders.
    ///
    /// # Parameters
    /// - `options`: New panel width, height, text, border, and highlights.
    pub fn set_options(&self, options: TextPanelOptions) {
        *self.options.lock().unwrap_or_else(PoisonError::into_inner) = options;
    }
}

impl Component for TextPanel {
    /// Renders panel rows within the allocated terminal width.
    ///
    /// # Parameters
    /// - `self`: Panel component to render.
    /// - `width`: Available terminal width in columns.
    ///
    /// # Returns
    /// Sanitized, clipped, theme-colored panel rows.
    fn render(&self, width: usize) -> Vec<String> {
        let options = self.options.lock().unwrap_or_else(PoisonError::into_inner);
        render_panel(&options, width)
    }

    /// Keeps the panel stateless between render passes.
    fn invalidate(&self) {}

    /// Exposes this component for upstream layout downcasting.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Converts one panel description into bounded, terminal-safe rows.
///
/// # Parameters
/// - `options`: Panel content and layout configuration.
/// - `available_width`: Width allocated by the containing layout.
///
/// # Returns
/// Rendered rows clipped to the panel's configured dimensions.
fn render_panel(options: &TextPanelOptions, available_width: usize) -> Vec<String> {
    let width = available_width.min(options.width);
    if width < 4 || options.max_height == 0 {
        return Vec::new();
    }
    let max_height = options.max_height.min(80);
    let colors = theme().colors;
    let inner = width.saturating_sub(if options.border { 4 } else { 0 });
    let has_border = options.border && max_height >= 2;
    let content_height = if has_border {
        max_height - 2
    } else {
        max_height
    };
    let content = std::iter::once(options.title.as_str())
        .filter(|title| !title.is_empty())
        .chain(options.lines.iter().map(String::as_str))
        .take(content_height)
        .map(|line| {
            let clean = sanitize(line);
            truncate_to_width(
                &highlighted(&clean, &options.highlights, &colors),
                inner,
                "",
            )
        })
        .collect::<Vec<_>>();
    if !has_border {
        return content;
    }
    let mut rows = Vec::with_capacity(max_height);
    rows.push(colors.dim.fg(&format!("┌{}┐", "─".repeat(width - 2))));
    for row in content {
        let padding = " ".repeat(inner.saturating_sub(visible_width(&row)));
        rows.push(format!(
            "{}{}{}",
            colors.dim.fg("│ "),
            row,
            colors.dim.fg(&format!("{padding} │"))
        ));
    }
    rows.push(colors.dim.fg(&format!("└{}┘", "─".repeat(width - 2))));
    rows
}

/// Removes terminal control sequences and all remaining control characters.
///
/// # Parameters
/// - `text`: Untrusted extension or host text.
///
/// # Returns
/// Plain text safe to include in rendered terminal rows.
fn sanitize(text: &str) -> String {
    strip_ansi(text)
        .chars()
        .filter(|character| !character.is_control())
        .take(MAX_PANEL_TEXT_CHARS)
        .collect()
}

/// Colors literal highlight matches and dims all other panel text.
///
/// # Parameters
/// - `text`: Sanitized panel line.
/// - `highlights`: Literal substrings to emphasize.
/// - `colors`: Active terminal theme colors.
///
/// # Returns
/// Theme-colored text preserving its visible content.
fn highlighted(text: &str, highlights: &[TextPanelHighlight], colors: &ThemeColors) -> String {
    let safe_highlights = highlights
        .iter()
        .take(MAX_HIGHLIGHTS)
        .map(|highlight| {
            (
                sanitize(&highlight.text)
                    .chars()
                    .take(MAX_HIGHLIGHT_CHARS)
                    .collect::<String>(),
                highlight.color,
            )
        })
        .filter(|(text, _)| !text.is_empty())
        .collect::<Vec<_>>();
    let mut result = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        let next = safe_highlights
            .iter()
            .filter_map(|(highlight, color)| {
                rest.find(highlight)
                    .map(|position| (position, highlight.as_str(), color))
            })
            .min_by_key(|(position, _, _)| *position);
        let Some((position, highlight, color)) = next else {
            result.push_str(&colors.dim.fg(rest));
            break;
        };
        result.push_str(&colors.dim.fg(&rest[..position]));
        let color = match color {
            TextPanelHighlightColor::Warning => colors.warning,
            TextPanelHighlightColor::Error => colors.error,
        };
        result.push_str(&color.fg(highlight));
        rest = &rest[position + highlight.len()..];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{TextPanel, TextPanelHighlight, TextPanelHighlightColor, TextPanelOptions};
    use crate::{strip_ansi, theme, visible_width, Component};

    /// Builds a compact panel configuration for rendering tests.
    ///
    /// # Returns
    /// A plain panel containing warning and error highlights.
    fn options() -> TextPanelOptions {
        TextPanelOptions {
            width: 240,
            max_height: 2,
            border: false,
            title: String::new(),
            lines: vec!["正常 \u{1b}[2JTTFT Err".to_owned()],
            highlights: vec![
                TextPanelHighlight {
                    text: "TTFT\u{1b}[2J".to_owned(),
                    color: TextPanelHighlightColor::Warning,
                },
                TextPanelHighlight {
                    text: "Err".to_owned(),
                    color: TextPanelHighlightColor::Error,
                },
            ],
        }
    }

    /// Clips panel rows and prevents content from injecting terminal controls.
    #[test]
    fn renders_sanitized_rows_with_active_theme_highlights() {
        let panel = TextPanel::new(options());
        let colors = theme().colors;
        let rendered = panel.render(80);
        assert_eq!(rendered.len(), 1);
        assert_eq!(strip_ansi(&rendered[0]), "正常 TTFT Err");
        assert!(!rendered[0].contains("\u{1b}[2J"));
        assert!(rendered[0].contains(&colors.warning.fg("TTFT")));
        assert!(rendered[0].contains(&colors.error.fg("Err")));
        for width in 4..32 {
            assert!(panel
                .render(width)
                .iter()
                .all(|row| visible_width(row) <= width));
        }
    }

    /// Uses replacement content on the next render pass.
    #[test]
    fn set_options_replaces_rendered_content() {
        let panel = TextPanel::new(options());
        let mut replacement = options();
        replacement.lines = vec!["updated".to_owned()];
        replacement.highlights.clear();
        panel.set_options(replacement);
        assert_eq!(strip_ansi(&panel.render(80)[0]), "updated");
    }

    /// Bounds rendered height even when a host supplies excessive content.
    #[test]
    fn clips_content_to_a_fixed_height_limit() {
        let mut config = options();
        config.border = true;
        config.max_height = usize::MAX;
        config.lines = (0..1_000).map(|index| format!("line {index}")).collect();
        let panel = TextPanel::new(config);
        let rows = panel.render(24);
        assert_eq!(rows.len(), 80);
        assert!(rows.iter().all(|row| visible_width(row) <= 24));
    }
}
