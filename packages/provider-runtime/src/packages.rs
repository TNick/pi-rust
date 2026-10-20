//! Read project package registries and discover their resource directories.

use std::path::{Component, Path, PathBuf};

use fs2::FileExt;
use rpi_extensions::{package_resource_dirs, ProjectPackage, ProjectPackageRegistry};

const REGISTRY_FILE: &str = "packages.json";
const LOCK_FILE: &str = "packages.lock";

/// Holds the exclusive project package registry mutation lock.
struct ProjectPackageLock(std::fs::File);

impl ProjectPackageLock {
    /// Lock the registry while a package mutation is in progress.
    ///
    /// # Parameters
    /// - `project_root`: Host-selected project root containing `.rpi`.
    ///
    /// # Returns
    /// The acquired package registry lock.
    ///
    /// # Errors
    /// Returns an error when the lock directory, file, or OS lock is
    /// unavailable.
    fn acquire(project_root: &Path) -> Result<Self, String> {
        let directory = project_root.join(".rpi");
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let path = directory.join(LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|error| error.to_string())?;
        file.lock_exclusive().map_err(|error| error.to_string())?;
        Ok(Self(file))
    }
}

impl Drop for ProjectPackageLock {
    /// Release the OS lock when the guard leaves scope.
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

/// Read the project package registry beneath a host-selected project root.
///
/// A missing registry is treated as an empty registry. This function parses
/// metadata only; callers remain responsible for validating package names,
/// authorizing roots, and deciding whether package code may be loaded.
///
/// # Parameters
/// - `project_root`: Root of the project that contains the `.rpi` directory.
///
/// # Returns
/// The parsed registry, or an empty registry when the file is absent.
///
/// # Errors
/// Returns an error when the registry cannot be read or parsed.
pub fn read_project_registry(project_root: &Path) -> Result<ProjectPackageRegistry, String> {
    let path = project_root.join(".rpi").join(REGISTRY_FILE);
    match std::fs::read_to_string(path) {
        Ok(contents) => ProjectPackageRegistry::from_json(&contents)
            .map_err(|error| format!("invalid project package registry: {error}")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(ProjectPackageRegistry::default())
        }
        Err(error) => Err(error.to_string()),
    }
}

/// Write project package registry metadata beneath a host-selected root.
///
/// This persists registry data only. It does not create, install, validate, or
/// authorize package artifacts.
///
/// # Parameters
/// - `project_root`: Root of the project that contains the `.rpi` directory.
/// - `registry`: Registry metadata to serialize.
///
/// # Returns
/// `Ok(())` after the staged registry file is renamed into place.
///
/// # Errors
/// Returns an error when the directory, serialization, write, or rename fails.
pub fn write_project_registry(
    project_root: &Path,
    registry: &ProjectPackageRegistry,
) -> Result<(), String> {
    let _lock = ProjectPackageLock::acquire(project_root)?;
    write_project_registry_unlocked(project_root, registry)
}

/// Persist project package registry metadata while the caller holds its lock.
///
/// # Parameters
/// - `project_root`: Root of the project that contains the `.rpi` directory.
/// - `registry`: Registry metadata to serialize.
///
/// # Returns
/// `Ok(())` after the staged registry file is renamed into place.
///
/// # Errors
/// Returns an error when the directory, serialization, write, or rename fails.
fn write_project_registry_unlocked(
    project_root: &Path,
    registry: &ProjectPackageRegistry,
) -> Result<(), String> {
    let directory = project_root.join(".rpi");
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(REGISTRY_FILE);
    let staged_path = directory.join(".packages.json.tmp");
    let bytes = serde_json::to_vec_pretty(registry).map_err(|error| error.to_string())?;
    std::fs::write(&staged_path, bytes).map_err(|error| error.to_string())?;
    std::fs::rename(staged_path, path).map_err(|error| error.to_string())
}

/// Find resource directories registered under the project's package root.
///
/// Returned paths are lexical joins, with invalid package keys rejected. The
/// caller must still authorize each resolved root before loading resources or
/// code; this function does not canonicalize or authorize filesystem roots.
///
/// # Parameters
/// - `project_root`: Root of the project containing `.rpi/packages`.
/// - `resource`: Relative resource directory name, such as `extensions`.
///
/// # Returns
/// Resource paths in registry order, including paths that do not exist.
///
/// # Errors
/// Returns an error when the registry cannot be read or parsed, package names
/// are invalid, or `resource` is not one relative path component.
pub fn project_package_resource_dirs(
    project_root: &Path,
    resource: &str,
) -> Result<Vec<PathBuf>, String> {
    let mut components = Path::new(resource).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err("resource must be one relative path component".to_string());
    }
    let registry = read_project_registry(project_root)?;
    if let Some(package) = registry
        .packages
        .iter()
        .find(|package| !valid_package_name(&package.name))
    {
        return Err(format!("invalid project package name `{}`", package.name));
    }
    Ok(package_resource_dirs(
        registry.package_roots(project_root.join(".rpi").join("packages")),
        resource,
    ))
}

/// Activate a built artifact as a project-local package.
///
/// The host selects `project_root` and owns its authorization policy. This
/// service validates the package key and updates the artifact plus registry;
/// it does not build, download, or load package code.
///
/// # Parameters
/// - `project_root`: Host-selected project root containing `.rpi/packages`.
/// - `package`: Registry record for the package being activated.
/// - `artifact`: Built dynamic library to copy into the package.
/// - `force`: Whether an existing artifact with the same filename may be
///   replaced.
///
/// # Returns
/// `Ok(())` after the artifact and registry have been written.
///
/// # Errors
/// Returns an error for invalid package metadata, an invalid artifact, or a
/// filesystem or serialization failure.
pub fn install_project_artifact(
    project_root: &Path,
    package: ProjectPackage,
    artifact: &Path,
    force: bool,
) -> Result<(), String> {
    if !valid_package_name(&package.name) {
        return Err(format!("invalid package name `{}`", package.name));
    }
    let _lock = ProjectPackageLock::acquire(project_root)?;
    let mut registry = read_project_registry(project_root)?;
    let metadata = std::fs::symlink_metadata(artifact)
        .map_err(|error| format!("could not inspect artifact {}: {error}", artifact.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "package artifact is not a regular file: {}",
            artifact.display()
        ));
    }
    let file = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "invalid artifact name".to_string())?;
    let package_root = project_root
        .join(".rpi")
        .join("packages")
        .join(&package.name);
    let extension_dir = package_root.join("extensions");
    std::fs::create_dir_all(&extension_dir)
        .map_err(|error| format!("could not create {}: {error}", extension_dir.display()))?;
    let target = extension_dir.join(file);
    if target.exists() && !force {
        return Err("package already exists; use --force".to_string());
    }
    let staged = extension_dir.join(format!(".{file}.tmp"));
    std::fs::copy(artifact, &staged)
        .map_err(|error| format!("could not stage package artifact: {error}"))?;
    if force {
        if let Err(error) = std::fs::remove_file(&target) {
            if error.kind() != std::io::ErrorKind::NotFound {
                let _ = std::fs::remove_file(&staged);
                return Err(format!(
                    "could not replace existing package artifact: {error}"
                ));
            }
        }
    }
    if let Err(error) = std::fs::rename(&staged, &target) {
        let _ = std::fs::remove_file(&staged);
        return Err(format!("could not activate package artifact: {error}"));
    }

    registry
        .packages
        .retain(|existing| existing.name != package.name);
    registry.packages.push(package);
    registry
        .packages
        .sort_by(|left, right| left.name.cmp(&right.name));
    write_project_registry_unlocked(project_root, &registry)
}

/// Remove a registered project package and its package directory.
///
/// The host selects `project_root` and owns its authorization policy. This
/// operation does not affect global packages or loaded sessions.
///
/// # Parameters
/// - `project_root`: Host-selected project root containing `.rpi/packages`.
/// - `name`: Registered package name to remove.
///
/// # Returns
/// `Ok(())` after the package directory and registry have been updated.
///
/// # Errors
/// Returns an error for an invalid name, absent registration, or filesystem or
/// registry failure.
pub fn uninstall_project_package(project_root: &Path, name: &str) -> Result<(), String> {
    if !valid_package_name(name) {
        return Err("invalid package name or arguments".to_string());
    }
    let _lock = ProjectPackageLock::acquire(project_root)?;
    let mut registry = read_project_registry(project_root)?;
    if !registry.packages.iter().any(|package| package.name == name) {
        return Err(format!("project package `{name}` is not installed"));
    }
    let package_dir = project_root.join(".rpi").join("packages").join(name);
    std::fs::remove_dir_all(&package_dir)
        .map_err(|error| format!("could not remove package: {error}"))?;
    registry.packages.retain(|package| package.name != name);
    write_project_registry_unlocked(project_root, &registry)
}

/// Check whether a package key is safe to use as one directory component.
///
/// # Parameters
/// - `name`: Package key to check.
///
/// # Returns
/// `true` when the key contains only ASCII letters, digits, dots, dashes, or
/// underscores.
fn valid_package_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}
