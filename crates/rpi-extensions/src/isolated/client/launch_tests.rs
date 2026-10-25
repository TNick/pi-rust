//! Exercises the Linux bubblewrap launcher against the host kernel.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use super::{
    LinuxBubblewrapLauncher, WorkerIsolation, WorkerLaunchSpec, WorkerLauncher,
    WorkerNetworkPolicy, WorkerSandboxPolicy, WorkerSecretPolicy,
};

/// Runs a bounded shell command inside the actual Linux namespace launcher.
#[test]
fn bubblewrap_launches_without_proc_and_hides_unbound_host_paths() {
    if !cfg!(target_os = "linux") || Command::new("bwrap").arg("--version").output().is_err() {
        return;
    }
    let launcher = LinuxBubblewrapLauncher;
    let spec = WorkerLaunchSpec {
        executable: PathBuf::from("/bin/sh"),
        args: vec![
            "-c".to_owned(),
            "test ! -e /proc/self/status && test ! -e /root && test \"$DECLARED_VALUE\" = allowed && test -z \"${HOST_SECRET_CANARY:-}\" && echo sandbox-ok".to_owned(),
        ],
        environment: BTreeMap::from([("DECLARED_VALUE".to_owned(), "allowed".to_owned())]),
        working_directory: Some(PathBuf::from("/tmp")),
        inherit_environment: false,
        sandbox: WorkerSandboxPolicy {
            isolation: WorkerIsolation::Sandboxed,
            filesystem: super::WorkerFilesystemPolicy {
                read_roots: ["/usr", "/bin", "/lib", "/lib64", "/etc"]
                    .into_iter()
                    .map(PathBuf::from)
                    .filter(|root| root.exists())
                    .collect(),
                write_roots: Vec::new(),
            },
            network: WorkerNetworkPolicy::Deny,
            secrets: WorkerSecretPolicy::DeclaredOnly,
        },
        startup_timeout: Duration::from_secs(10),
        request_timeout: Duration::from_secs(10),
    };
    let mut command = launcher.command(&spec).expect("build bubblewrap command");
    command.env("HOST_SECRET_CANARY", "must-not-inherit");
    let output = command.output().expect("start bubblewrap process");
    if !output.status.success()
        && String::from_utf8_lossy(&output.stderr).contains("Operation not permitted")
    {
        return;
    }
    assert!(
        output.status.success(),
        "bubblewrap failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("sandbox-ok"));
}
