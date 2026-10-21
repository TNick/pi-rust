//! Resolves host-provided environment templates in shared model configuration.

use std::collections::BTreeMap;

/// Resolves `$NAME` and `${NAME}` references without owning environment access.
///
/// Repeated variables are interpolated in order. An absent variable makes the
/// complete value unresolved. `$$` and `$!` escape the following character;
/// malformed braced names and bare dollar signs remain literal.
///
/// # Parameters
/// - `value`: Literal or environment template from host configuration.
/// - `overlay`: Optional scoped environment values that take precedence.
/// - `environment_value`: Host callback that reads one process environment
///   variable without exposing the complete environment to this crate.
///
/// # Returns
/// The interpolated value, or `None` when a referenced variable is absent.
pub fn resolve_environment_template(
    value: &str,
    overlay: Option<&BTreeMap<String, String>>,
    mut environment_value: impl FnMut(&str) -> Option<String>,
) -> Option<String> {
    let mut output = String::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let Some(offset) = value[index..].find('$') else {
            output.push_str(&value[index..]);
            break;
        };
        let dollar = index + offset;
        output.push_str(&value[index..dollar]);
        let after = dollar + 1;
        match bytes.get(after).copied() {
            Some(b'$' | b'!') => {
                output.push(char::from(bytes[after]));
                index = after + 1;
            }
            Some(b'{') => {
                let Some(end_offset) = value[after + 1..].find('}') else {
                    output.push('$');
                    index = after;
                    continue;
                };
                let end = after + 1 + end_offset;
                let name = &value[after + 1..end];
                if is_environment_name(name) {
                    let resolved = overlay
                        .and_then(|values| values.get(name).cloned())
                        .or_else(|| environment_value(name))?;
                    output.push_str(&resolved);
                } else {
                    output.push_str(&value[dollar..=end]);
                }
                index = end + 1;
            }
            _ => {
                if let Some(name) = environment_name_prefix(&value[after..]) {
                    let resolved = overlay
                        .and_then(|values| values.get(name).cloned())
                        .or_else(|| environment_value(name))?;
                    output.push_str(&resolved);
                    index = after + name.len();
                } else {
                    output.push('$');
                    index = after;
                }
            }
        }
    }
    Some(output)
}

/// Checks whether a braced environment variable name is valid.
///
/// # Parameters
/// - `name`: Candidate name without `${` and `}`.
///
/// # Returns
/// Whether the name follows the portable ASCII environment-name grammar.
fn is_environment_name(name: &str) -> bool {
    let mut characters = name.chars();
    matches!(
        characters.next(),
        Some(character) if character.is_ascii_alphabetic() || character == '_'
    ) && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Returns the longest valid variable-name prefix at the start of a string.
///
/// # Parameters
/// - `value`: Text immediately following an unbraced dollar sign.
///
/// # Returns
/// The variable-name prefix, or `None` when no valid prefix starts here.
fn environment_name_prefix(value: &str) -> Option<&str> {
    let mut characters = value.char_indices();
    match characters.next() {
        Some((_, character)) if character.is_ascii_alphabetic() || character == '_' => {}
        _ => return None,
    }
    let end = characters
        .find(|(_, character)| !(character.is_ascii_alphanumeric() || *character == '_'))
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    Some(&value[..end])
}
