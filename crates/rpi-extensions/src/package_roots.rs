//! Resolve resource directories beneath caller-selected extension packages.

use std::path::{Path, PathBuf};

/// Resolve one resource directory beneath each supplied package root.
///
/// The caller owns package discovery and policy. This helper preserves the
/// supplied order and returns paths whether or not they currently exist.
///
/// # Parameters
/// - `package_roots`: Package directories selected by the caller.
/// - `resource`: Relative resource directory name, such as `extensions`.
///
/// # Returns
/// One resource path per package root, in the supplied order.
pub fn package_resource_dirs<I, P>(package_roots: I, resource: &str) -> Vec<PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    package_roots
        .into_iter()
        .map(|root| root.as_ref().join(resource))
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for caller-owned package resource path resolution.

    use super::package_resource_dirs;
    use std::path::PathBuf;

    /// Keeps output order aligned with the caller-supplied package roots.
    #[test]
    fn preserves_package_root_order() {
        let roots = [PathBuf::from("first"), PathBuf::from("second")];

        assert_eq!(
            package_resource_dirs(roots, "extensions"),
            vec![
                PathBuf::from("first/extensions"),
                PathBuf::from("second/extensions"),
            ]
        );
    }

    /// Includes paths even when the package roots do not exist.
    #[test]
    fn retains_nonexistent_package_roots() {
        let roots = [PathBuf::from("absent-package")];

        assert_eq!(
            package_resource_dirs(roots, "extensions"),
            vec![PathBuf::from("absent-package/extensions")]
        );
    }

    /// Returns no paths when the caller supplies no package roots.
    #[test]
    fn accepts_empty_package_roots() {
        let roots: [PathBuf; 0] = [];

        assert!(package_resource_dirs(roots, "extensions").is_empty());
    }
}
