# HiveOS integration

## Persistent paths

HiveOS replaces the complete miner package during an upgrade or rollback. Keryx therefore stores funds-critical mutable data outside that package:

```text
/hive/miners/custom/keryx-miner-state/escrow.key
/hive/miners/custom/keryx-miner-state/escrow_state.json
```

The directory is mode `0700`; both files are mode `0600`. `h-run.sh` validates and migrates package-local files before starting the miner. A valid existing durable file takes precedence, making migration idempotent. Invalid durable or legacy data stops migration without replacing either copy.

Back up both durable files together before an upgrade or recovery. If validation fails, restore the matching pair from backup. Do not delete `escrow.key` to regenerate it because the original key controls existing rewards.

## Rollback-safe upgrade

An old Keryx package understands the escrow path flags but defaults to files in its replaceable working directory. Before changing the Install URL, persist these exact flags in the HiveOS Flight Sheet **Extra config**:

```text
--escrow-key-file /hive/miners/custom/keryx-miner-state/escrow.key --escrow-state-file /hive/miners/custom/keryx-miner-state/escrow_state.json
```

Use the `pre-hive-upgrade.sh` bundled with the exact target release. Do not pipe a separately downloaded script into a shell. Copy the bundled script to a temporary path, then run it with the current package directory explicitly:

```bash
install -m 0700 /path/to/unpacked-target-release/pre-hive-upgrade.sh /tmp/keryx-pre-hive-upgrade.sh
CURRENT_KERYX_DIR=/hive/miners/custom/name-of-currently-installed-keryx-package
test -f "$CURRENT_KERYX_DIR/h-manifest.conf" || exit 1
bash /tmp/keryx-pre-hive-upgrade.sh --install-dir "$CURRENT_KERYX_DIR"
```

Replace `name-of-currently-installed-keryx-package` with the directory containing the currently running package, not the unpacked target release. The first run stops the current miner and performs verified migration. It cannot report the upgrade safe unless a valid durable key exists. If persistent flags are missing, it prints the exact Extra config and exits without declaring the upgrade safe. Add those flags, apply the Flight Sheet while the old package is still installed, then run the command again. The second run must report that migration and rollback flags are verified. It sources `/hive-config/wallet.conf` in an isolated Bash process for verification and never edits it.

After verification, change only the Install URL and preserve Extra config. The new package and an unchanged old rollback package will then read and write the same durable files. If HiveOS reports `Already installed` instead of replacing the package, force the requested install through the normal HiveOS custom-miner controls.

Custom `--escrow-key-file` and `--escrow-state-file` values remain authoritative in `h-config.sh`, using either `--flag path` or `--flag=path`. Rollback is guaranteed only when both persistent values point outside the package and are retained across the Flight Sheet change.

## Package naming

Use the release archive format:

```text
keryx-miner-v<version>_OPoI_hiveos.tar.gz
```

This gives HiveOS the stable miner name `keryx-miner`. Models remain in `/hive/miners/custom/models`; escrow data must remain in the separate state directory above.
