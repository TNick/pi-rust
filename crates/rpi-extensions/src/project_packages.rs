//! Parse project package records and resolve their host-selected roots.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Describes one project-local extension package registration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectPackage {
    /// Registry key and package directory name.
    pub name: String,
    /// Package version recorded by the host.
    pub version: String,
    /// Package source recorded by the host.
    pub source: String,
    /// Cargo package name used to build the extension.
    pub cargo_package: String,
    /// Built extension artifact names.
    #[serde(default)]
    pub artifacts: Vec<String>,
}

/// Stores project-local extension package registrations.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectPackageRegistry {
    /// Registered project packages in host-defined order.
    #[serde(default)]
    pub packages: Vec<ProjectPackage>,
}

impl ProjectPackageRegistry {
    /// Parses project package registry JSON.
    ///
    /// This parses data only. It does not validate package names or authorize
    /// package roots for loading.
    ///
    /// # Parameters
    /// - `json`: Serialized project package registry.
    ///
    /// # Returns
    /// The parsed registry.
    ///
    /// # Errors
    /// Returns the JSON parsing error when the input is malformed.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Resolves registered package directories beneath a host-selected root.
    ///
    /// Returned paths are lexical joins only. They are not checked for
    /// traversal, canonicalized, checked for existence, or authorized. Hosts
    /// must validate and authorize every result before loading code from it.
    ///
    /// # Parameters
    /// - `packages_root`: Root selected by the host for package directories.
    ///
    /// # Returns
    /// Package directory paths in registry order.
    pub fn package_roots(&self, packages_root: impl AsRef<Path>) -> Vec<PathBuf> {
        let packages_root = packages_root.as_ref();
        self.packages
            .iter()
            .map(|package| packages_root.join(&package.name))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for project package registry parsing and root resolution.

    use super::{ProjectPackage, ProjectPackageRegistry};
    use std::path::PathBuf;

    /// Parses registry records and defaults the optional artifact list.
    #[test]
    fn parses_project_package_registry_json() {
        let registry = ProjectPackageRegistry::from_json(
            r#"{"packages":[{"name":"example","version":"1.0.0","source":"local","cargo_package":"example"}]}"#,
        )
        .expect("valid project package registry");

        assert_eq!(
            registry.packages,
            vec![ProjectPackage {
                name: "example".to_owned(),
                version: "1.0.0".to_owned(),
                source: "local".to_owned(),
                cargo_package: "example".to_owned(),
                artifacts: Vec::new(),
            }]
        );
    }

    /// Rejects malformed registry JSON with the parser error.
    #[test]
    fn rejects_malformed_project_package_registry_json() {
        assert!(ProjectPackageRegistry::from_json("not json").is_err());
    }

    /// Resolves package roots beneath the host-selected project root.
    #[test]
    fn resolves_project_package_roots_in_registry_order() {
        let registry = ProjectPackageRegistry {
            packages: ["first", "second"]
                .into_iter()
                .map(|name| ProjectPackage {
                    name: name.to_owned(),
                    version: "1.0.0".to_owned(),
                    source: "local".to_owned(),
                    cargo_package: name.to_owned(),
                    artifacts: Vec::new(),
                })
                .collect(),
        };

        assert_eq!(
            registry.package_roots("project/.rpi/packages"),
            vec![
                PathBuf::from("project/.rpi/packages/first"),
                PathBuf::from("project/.rpi/packages/second"),
            ]
        );
    }

    /// Returns no package roots for an empty registry.
    #[test]
    fn resolves_no_roots_for_empty_registry() {
        assert!(ProjectPackageRegistry::default()
            .package_roots("project")
            .is_empty());
    }
}
