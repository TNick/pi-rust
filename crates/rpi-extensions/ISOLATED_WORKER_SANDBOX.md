# Isolated plugin worker sandbox contract

The isolated worker launcher separates two concepts:

- **Process containment** — the worker is a child process with piped stdio and a
  supervised process tree. Environment clearing and process-tree lifetime are
  useful hygiene, but they do not restrict what the worker can open.
- **OS sandbox** — the launcher asks the operating system to deny filesystem,
  network, or secret access that the host did not grant.

A launch spec requesting `WorkerIsolation::Sandboxed` is rejected unless the
launcher reports that it enforces all three restrictions:

```rust
WorkerSandboxGuarantees {
    filesystem: true,
    network: true,
    secrets: true,
    platform: "…",
}
```

`IsolatedWorker::start` checks `WorkerLauncher::guarantees` before spawning and
returns `WorkerError::Sandbox` when the requested isolation is not enforced.
`HostProcessLauncher` also rejects sandboxed specs directly. There is no silent
fallback from a sandboxed request to an ordinary child process.

## Host-selected policy

`WorkerSandboxPolicy` carries:

- `filesystem.read_roots` — absolute paths the worker may read.
- `filesystem.write_roots` — absolute paths the worker may write.
- `network` — `Deny` or `Allow`.
- `secrets` — `DeclaredOnly` (clear host environment, pass only
  `WorkerLaunchSpec.environment`) or `DenyAll` (clear all environment).
- `isolation` — `Unisolated` or `Sandboxed`.

`WorkerLaunchSpec.inherit_environment` must be false for sandboxed execution;
the Linux launcher rejects it. Hosts must not place credentials in the declared
worker environment.

## Platform enforcement

### Linux — `LinuxBubblewrapLauncher`

Implemented with `bwrap` and reports full guarantees.

Enforced:

- filesystem: only the declared absolute read roots are mounted read-only and
  write roots read-write; `/proc`, `/dev`, and a private `/tmp` are synthetic
  sandbox mounts;
- network: `--unshare-all` denies network, `--share-net` re-attaches the host
  network only for `WorkerNetworkPolicy::Allow`;
- secrets: `--clearenv` removes the host environment; only
  `WorkerSecretPolicy::DeclaredOnly` values are passed with `--setenv`;
- privilege/namespace separation: user, PID, IPC, UTS, cgroup, and network
  namespaces via `--unshare-all`, plus `--cap-drop ALL`;
- lifecycle: `--die-with-parent` and a new session, in addition to the host's
  process-tree supervision.

Failure behavior: non-Linux hosts, inherited host environments, relative or
empty roots are rejected before launch. A missing/incompatible `bwrap`, denied
user namespaces, or an unrepresentable filesystem layout fails the launch. The
launcher does not fall back to `HostProcessLauncher`.

Known limits: `bwrap` must be installed and permitted by host policy; the
kernel must allow the requested namespaces; the launcher does not add a seccomp
syscall filter, device policy beyond the synthetic `/dev`, or CPU/memory
resource limits; declared environment values are passed as `--setenv` argv and
may be visible to host process observers, so hosts must not put secrets there.
`--share-net` deliberately reattaches the host network namespace when the host
selects `WorkerNetworkPolicy::Allow`.

### macOS

No built-in launcher is provided. `HostProcessLauncher` fails closed for
sandboxed requests. A host must inject a `WorkerLauncher` backed by the
platform mechanism it trusts (for example a `sandbox-exec` profile or an
external sandbox runner) and report truthful `WorkerSandboxGuarantees`.
Without that injection, macOS cannot satisfy `WorkerIsolation::Sandboxed`.

### Windows

No built-in launcher is provided. `HostProcessLauncher` fails closed for
sandboxed requests. A host must inject a launcher built on an OS isolation
primitive that it has reviewed (for example AppContainer and a restricted job
object) and report truthful guarantees. Job-object lifetime supervision alone
is not a filesystem, network, or secret sandbox and must not be reported as
one.

## Trust boundary

`WorkerSandboxGuarantees` is a host-injected launcher contract, not a
verification of the launcher implementation. A launcher that reports guarantees
it does not enforce defeats the boundary. Hosts should construct launchers from
reviewed platform code and must not enable plugin activation merely because a
launcher type is available.

The worker protocol, plugin loading, and tool execution remain in the worker
process; the host-side capability allowlist governs host-mediated operations.
