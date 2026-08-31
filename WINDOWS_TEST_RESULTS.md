# Windows IPFS Test Results

**Status:** Passed  
**Date:** 2026-08-31  
**Branch:** `fix/windows-kubo-repo-path`  
**Revision:** `114e43b79075de851a62aa11a7bc0ee57a6f1974`

## Environment

- Windows 10 Home 22H2, 64-bit AMD64
- Rust 1.98.0 (`stable-x86_64-pc-windows-msvc`)
- Visual Studio Build Tools 2022, MSVC 19.44.35228
- Windows SDK 10.0.19041.0
- CUDA Toolkit 12.4.1, nvcc 12.4.131
- protoc 36.0
- CMake 4.4.3

## Command

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
set KERYX_LLAMA_SKIP=1
set CUDARC_CUDA_VERSION=12040
cd /d C:\Users\agentops\keryx-build-validation
cargo test --bin keryx-miner ipfs::tests
```

## Result

```text
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 56 filtered out; finished in 2.04s
```

- Cargo exit code: `0`
- Clean build duration: 3 minutes 50 seconds
- Total command duration: approximately 3 minutes 53 seconds
- All 115 staged source files remained byte-identical before and after the test.
- No NVIDIA display driver was installed or required.
- The miner was not started and no mining network connection was made.

## Native Windows Live Kubo Validation

**Status:** Passed  
**Date:** 2026-08-31  
**Working directory:** `C:\Users\agentops\keryx-live-validation`  
**Revision:** `114e43b79075de851a62aa11a7bc0ee57a6f1974`

The temporary ignored-test harness called the real
`src/ipfs.rs::ensure_daemon` implementation. It was not committed and was
removed after validation.

### Prerequisites

- Port 5001 was free before every managed-start case.
- No existing Kubo process was used as evidence.
- Kubo 0.41.0 Windows AMD64 was downloaded from the official IPFS
  distribution and verified with the published SHA-512:

```text
c0d80cc3261c6ab4c47f477f393b1c03322c5dd89a2b598f95568eb4bbac6d85bc6ca177796da253fe75fb05188bc9f44d78b853a73fad51298d414f390c6699
```

- The committed focused tests were reconfirmed before installing the
  temporary harness:

```cmd
cargo test --bin keryx-miner ipfs::tests -- --test-threads=1
```

```text
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 56 filtered out
```

### Case A: explicit `IPFS_PATH`

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_explicit_ipfs_path -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- `HOME`: unset
- `USERPROFILE`: unset
- Expected and actual repository:
  `C:\Users\agentops\keryx-live-validation\live-test-data\explicit-repo`
- Repository `config`: present
- Kubo executable:
  `C:\Users\agentops\keryx-live-validation\live-test-data\verified-kubo\kubo\ipfs.exe`
- Kubo version: `0.41.0`
- API ready: yes
- Kubo PID: `2860`
- Log:
  `C:\Users\agentops\keryx-live-validation\.keryx\kubo.log`
- Shutdown succeeded and port 5001 closed.
- No fallback repository was created under the staging directory.

### Case B: default `USERPROFILE\.ipfs`

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_userprofile_default -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- `HOME`: unset
- `IPFS_PATH`: unset
- `USERPROFILE`:
  `C:\Users\agentops\keryx-live-validation\live-test-data\default-profile`
- Expected and actual repository: `default-profile\.ipfs`
- Repository `config`: present
- Kubo executable/version: verified Kubo `0.41.0`
- API ready: yes
- Kubo PID: `1504`
- Log: `default-profile\.keryx\kubo.log`
- Shutdown succeeded and port 5001 closed.
- No repository was created under the staging current directory.

### Case C: adjacent `ipfs.exe`

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_adjacent_binary -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- Kubo was removed from the process `PATH`.
- Expected and actual repository: `adjacent-profile\.ipfs`
- Resolved executable:
  `C:\Users\agentops\keryx-live-validation\target\debug\deps\ipfs.exe`
- Kubo version: `0.41.0`
- API ready: yes
- Kubo PID: `1636`
- Shutdown succeeded and port 5001 closed.
- The adjacent fixture was removed after the daemon released its executable
  lock.

### Case D: automatic ZIP download

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_downloaded_binary -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- Kubo was absent from the process `PATH`, and no adjacent executable existed
  before the call.
- Expected and actual repository: `download-profile\.ipfs`
- Download URL:
  `https://dist.ipfs.tech/kubo/v0.43.0/kubo_v0.43.0_windows-amd64.zip`
- Archive: `kubo_v0.43.0_windows-amd64.zip`
- Extracted executable:
  `C:\Users\agentops\keryx-live-validation\target\debug\deps\ipfs.exe`
- Downloaded Kubo version: `0.43.0`
- API ready: yes
- Kubo PID: `2456`
- Shutdown succeeded and port 5001 closed.
- The extracted executable was removed.

This case validates the production automatic-download behavior, not artifact
authenticity. Production selected the current upstream release rather than the
0.41.0 fallback and does not independently verify its checksum.

### Case E: recovery and repository reuse

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_restart_existing_repo -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- Repository: `restart-profile\.ipfs`
- Kubo executable/version: verified Kubo `0.41.0`
- First Kubo PID: `6640`
- Second Kubo PID: `7256`
- Both API readiness checks succeeded.
- The existing repository and `config` were reused.
- Exactly one daemon owned port 5001 after restart.
- Both shutdowns succeeded and port 5001 closed.

### Case F: remote endpoint remains unmanaged

```cmd
cargo test --bin keryx-miner ipfs::tests::windows_live_remote_unmanaged -- --ignored --exact --nocapture --test-threads=1
```

- Exit code: `0`
- Endpoint: `http://192.0.2.1:5001`
- Result: expected `remote endpoints are not auto-managed` error
- No Kubo process started.
- No repository, log, archive, or adjacent executable was created.
- Observed probe duration: approximately 21.02 seconds.

The remote case remained side-effect-free, but the observed Windows network
timeout exceeded the nominal two-second production probe timeout.

### Cleanup and integrity proof

```text
HEAD=114e43b79075de851a62aa11a7bc0ee57a6f1974
working tree clean
port 5001 owners=0
ipfs.exe processes=0
live-test-data exists=False
adjacent ipfs.exe fixture exists=False
staging .keryx exists=False
```

The temporary harness, repositories, logs, runner scripts, verified Kubo
fixture, downloaded executable, and source-transfer bundle were removed.
`src/ipfs.rs` was restored with `git restore`, and its final diff was empty.

Direct cloning on the Windows host failed because that host lacked GitHub SSH
authorization. The exact branch was therefore staged from a temporary Git
bundle created from the clean local revision; the bundle was removed after
validation.
