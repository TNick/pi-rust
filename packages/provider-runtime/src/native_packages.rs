//! Generic native package registry and version lifecycle primitives.

use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{de::DeserializeOwned, Serialize};

const LOCK_FILE: &str = "native-packages.lock";

/// Represents whether native package metadata is absent or valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Registry<T> {
    /// No registry exists yet.
    Missing,
    /// The registry exists and contains these records.
    Valid(Vec<T>),
}

impl<T> Registry<T> {
    /// Returns the stored records, treating a missing registry as empty.
    pub fn into_records(self) -> Vec<T> {
        match self {
            Self::Missing => Vec::new(),
            Self::Valid(records) => records,
        }
    }
}

/// Holds the exclusive lock for a host-selected package registry path.
pub struct MutationLock(std::fs::File);

impl MutationLock {
    /// Acquires an exclusive lock adjacent to the registry file.
    ///
    /// # Parameters
    /// - `metadata_path`: Host-selected package metadata path.
    ///
    /// # Returns
    /// The lock guard, released when dropped.
    ///
    /// # Errors
    /// Returns an error when the directory, lock file, or OS lock is unavailable.
    pub fn acquire(metadata_path: &Path) -> Result<Self, String> {
        let parent = metadata_path
            .parent()
            .ok_or_else(|| "native package metadata has no parent".to_string())?;
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let path = parent.join(LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|error| error.to_string())?;
        file.try_lock_exclusive()
            .map_err(|error| error.to_string())?;
        Ok(Self(file))
    }
}

impl Drop for MutationLock {
    /// Releases the OS lock.
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

/// Reads and validates a package registry without hiding file errors.
///
/// # Parameters
/// - `path`: Host-selected registry file.
///
/// # Returns
/// Missing or parsed registry state.
///
/// # Errors
/// Returns an error when the file cannot be read or decoded.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Registry<T>, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Registry::Missing),
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Registry::Valid)
        .map_err(|error| format!("invalid package registry {}: {error}", path.display()))
}

/// Atomically writes package registry records beneath a host-selected path.
///
/// Callers that coordinate artifact changes should hold [`MutationLock`] and
/// stage this write in the same transaction as their artifacts.
///
/// # Parameters
/// - `path`: Host-selected registry file.
/// - `records`: Registry records to serialize.
///
/// # Returns
/// `Ok(())` when the replacement file is activated.
///
/// # Errors
/// Returns an error when serialization or filesystem operations fail.
pub fn write<T: Serialize>(path: &Path, records: &[T]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "native package metadata has no parent".to_string())?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let staged = temporary_path(path);
    let bytes = serde_json::to_vec_pretty(records).map_err(|error| error.to_string())?;
    std::fs::write(&staged, bytes).map_err(|error| error.to_string())?;
    let backup = temporary_path(path);
    let had_previous = match std::fs::rename(path, &backup) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            let _ = std::fs::remove_file(&staged);
            return Err(error.to_string());
        }
    };
    if let Err(error) = std::fs::rename(&staged, path) {
        let _ = std::fs::remove_file(&staged);
        let restoration = if had_previous {
            std::fs::rename(&backup, path).map_err(|restore| restore.to_string())
        } else {
            Ok(())
        };
        return match restoration {
            Ok(()) => Err(error.to_string()),
            Err(restore) => Err(format!("{error}; registry restoration failed: {restore}")),
        };
    }
    if had_previous {
        std::fs::remove_file(backup).map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Determines whether a requested package version may replace an installed one.
///
/// # Parameters
/// - `name`: Package name used in diagnostics.
/// - `current`: Installed version, if any.
/// - `requested`: Candidate version.
/// - `force`: Whether equal versions and downgrades are allowed.
///
/// # Returns
/// `true` when installation should proceed.
///
/// # Errors
/// Returns an error for malformed versions or an unforced downgrade.
pub fn should_install_version(
    name: &str,
    current: Option<&str>,
    requested: &str,
    force: bool,
) -> Result<bool, String> {
    let Some(current) = current.filter(|_| !force) else {
        return Ok(true);
    };
    let parse = |version: &str| {
        semver::Version::parse(version).map_err(|_| {
            format!("cannot compare version `{version}` for `{name}`; use --force to replace it")
        })
    };
    match parse(requested)?.cmp_precedence(&parse(current)?) {
        std::cmp::Ordering::Greater => Ok(true),
        std::cmp::Ordering::Equal => Ok(false),
        std::cmp::Ordering::Less => Err(format!(
            "{name} {current} is installed; use --force to downgrade to {requested}"
        )),
    }
}

/// Allocates a sibling path for staged registry contents.
///
/// # Parameters
/// - `path`: Registry path whose parent will contain the temporary file.
///
/// # Returns
/// A unique sibling path candidate.
fn temporary_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    parent.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()))
}

#[cfg(test)]
mod tests {
    //! Tests for generic native package registry lifecycle primitives.

    use super::{read, should_install_version, write, Registry};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
    struct Record {
        name: String,
    }

    /// Round trips a registry and distinguishes a missing file.
    #[test]
    fn registry_round_trip_and_missing_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("registry.json");
        assert!(matches!(read::<Record>(&path).unwrap(), Registry::Missing));
        let records = vec![Record {
            name: "demo".into(),
        }];
        write(&path, &records).unwrap();
        assert_eq!(read(&path).unwrap(), Registry::Valid(records));
    }

    /// Applies semver update and downgrade policy consistently.
    #[test]
    fn version_policy_allows_upgrade_and_rejects_downgrade() {
        assert!(should_install_version("demo", Some("1.0.0"), "1.1.0", false).unwrap());
        assert!(!should_install_version("demo", Some("1.0.0"), "1.0.0", false).unwrap());
        assert!(should_install_version("demo", Some("1.1.0"), "1.0.0", false).is_err());
        assert!(should_install_version("demo", Some("1.1.0"), "1.0.0", true).unwrap());
    }
}
