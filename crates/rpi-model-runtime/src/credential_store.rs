//! Path-parameterized persistence for host-owned credential stores.

use std::path::{Path, PathBuf};

use fs2::FileExt;
use thiserror::Error;

use crate::{AuthStore, Credential};

/// Error returned by the reusable credential store persistence API.
#[derive(Debug, Error)]
pub enum CredentialStoreError {
    /// The credential file could not be read or written.
    #[error("could not access credential store {path}: {source}")]
    Io {
        /// Credential file or lock path involved in the operation.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// The credential file was not valid JSON for an auth store.
    #[error("invalid credential store JSON in {path}: {source}")]
    Json {
        /// Credential file path containing invalid data.
        path: PathBuf,
        /// Underlying JSON decoding failure.
        #[source]
        source: serde_json::Error,
    },
}

/// Reads a host-selected credential file, returning an empty store if absent.
///
/// # Parameters
/// - `path`: Exact credential file path selected by the host.
///
/// # Returns
/// The decoded auth store or an empty store when the file does not exist.
///
/// # Errors
/// Returns an error for IO failures or malformed JSON.
pub fn read_auth_store(path: &Path) -> Result<AuthStore, CredentialStoreError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(AuthStore::new()),
        Err(source) => {
            return Err(CredentialStoreError::Io {
                path: path.to_path_buf(),
                source,
            })
        }
    };
    serde_json::from_slice(&bytes).map_err(|source| CredentialStoreError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Atomically writes a host-selected credential file with owner-only mode.
///
/// The caller selects the path; this API does not choose a config root or
/// environment policy. On Unix, newly created parent directories use mode
/// `0700` and the staged credential file uses mode `0600`.
///
/// # Parameters
/// - `path`: Exact credential file path selected by the host.
/// - `store`: Credential mapping to persist.
///
/// # Returns
/// `Ok(())` after the staged contents replace the file.
///
/// # Errors
/// Returns an error when locking, serialization, or filesystem operations fail.
pub fn write_auth_store(path: &Path, store: &AuthStore) -> Result<(), CredentialStoreError> {
    with_store_lock(path, || write_unlocked(path, store))
}

/// Upserts one credential while holding the store mutation lock.
///
/// # Parameters
/// - `path`: Exact credential file path selected by the host.
/// - `provider_id`: Stable provider identity.
/// - `credential`: Credential value to store.
///
/// # Returns
/// `Ok(())` after the updated store is persisted.
///
/// # Errors
/// Returns an error for lock, IO, or JSON failures.
pub fn upsert_auth_credential(
    path: &Path,
    provider_id: &str,
    credential: Credential,
) -> Result<(), CredentialStoreError> {
    with_store_lock(path, || {
        let mut store = read_unlocked(path)?;
        store.insert(provider_id.to_owned(), credential);
        write_unlocked(path, &store)
    })
}

/// Deletes one credential and persists only when it was present.
///
/// # Parameters
/// - `path`: Exact credential file path selected by the host.
/// - `provider_id`: Stable provider identity to remove.
///
/// # Returns
/// `true` when a credential was removed and persisted.
///
/// # Errors
/// Returns an error for lock, IO, or JSON failures.
pub fn delete_auth_credential(
    path: &Path,
    provider_id: &str,
) -> Result<bool, CredentialStoreError> {
    with_store_lock(path, || {
        let mut store = read_unlocked(path)?;
        if store.remove(provider_id).is_some() {
            write_unlocked(path, &store)?;
            Ok(true)
        } else {
            Ok(false)
        }
    })
}

/// Serializes credential mutations for one host-selected file.
fn with_store_lock<T>(
    path: &Path,
    operation: impl FnOnce() -> Result<T, CredentialStoreError>,
) -> Result<T, CredentialStoreError> {
    let parent = path.parent().ok_or_else(|| CredentialStoreError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no parent"),
    })?;
    ensure_private_directory(parent).map_err(|source| CredentialStoreError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let lock_path = parent.join(format!(
        ".{}.lock",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| CredentialStoreError::Io {
            path: lock_path.clone(),
            source,
        })?;
    lock.lock_exclusive()
        .map_err(|source| CredentialStoreError::Io {
            path: lock_path,
            source,
        })?;
    operation()
}

/// Reads credential data without acquiring a second lock.
fn read_unlocked(path: &Path) -> Result<AuthStore, CredentialStoreError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(AuthStore::new()),
        Err(source) => {
            return Err(CredentialStoreError::Io {
                path: path.to_path_buf(),
                source,
            })
        }
    };
    serde_json::from_slice(&bytes).map_err(|source| CredentialStoreError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Replaces the credential file while the caller holds its mutation lock.
fn write_unlocked(path: &Path, store: &AuthStore) -> Result<(), CredentialStoreError> {
    let parent = path.parent().ok_or_else(|| CredentialStoreError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no parent"),
    })?;
    ensure_private_directory(parent).map_err(|source| CredentialStoreError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let staged = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        uuid::Uuid::new_v4().simple()
    ));
    let json = serde_json::to_vec_pretty(store).expect("AuthStore serializes");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&staged)
        .map_err(|source| CredentialStoreError::Io {
            path: staged.clone(),
            source,
        })?;
    if let Err(source) = std::io::Write::write_all(&mut file, &json).and_then(|()| file.sync_all())
    {
        let _ = std::fs::remove_file(&staged);
        return Err(CredentialStoreError::Io {
            path: staged,
            source,
        });
    }
    drop(file);
    std::fs::rename(&staged, path).map_err(|source| {
        let _ = std::fs::remove_file(&staged);
        CredentialStoreError::Io {
            path: path.to_path_buf(),
            source,
        }
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Creates a credential directory with private Unix permissions.
fn ensure_private_directory(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests for path-parameterized credential persistence.

    use super::{delete_auth_credential, read_auth_store, upsert_auth_credential};
    use crate::Credential;

    /// Reads missing stores, upserts credentials, and removes entries.
    #[test]
    fn credential_store_lifecycle_round_trips_without_path_policy() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("host-selected-auth.json");
        assert!(read_auth_store(&path).unwrap().is_empty());
        upsert_auth_credential(
            &path,
            "provider",
            Credential::ApiKey {
                key: Some("secret".to_owned()),
                env: None,
            },
        )
        .unwrap();
        assert!(matches!(
            read_auth_store(&path).unwrap().get("provider"),
            Some(Credential::ApiKey { key: Some(value), .. }) if value == "secret"
        ));
        assert!(delete_auth_credential(&path, "provider").unwrap());
        assert!(!delete_auth_credential(&path, "provider").unwrap());
    }
}
