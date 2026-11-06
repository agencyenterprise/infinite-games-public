#!/usr/bin/env bash
# Pull infinite-games outputs from the RunPod box(es) into this repo's outputs/. Adapted from casper's
# scripts/sync_casper_outputs.sh; same contract.
#
# THE CONTRACT. This script COLLECTS. It may pull new files and it may overwrite a local file with a newer one from a host. It
# must NEVER remove anything local: no --delete, no --delete-excluded, no --remove-source-files, and no rm in any executable
# line (tests/test_repo.py asserts this). A local file that is not on the host is left alone, because the local copy is the
# accumulated record across hosts that come and go.
#
# The whole outputs/ tree is pulled in one rsync per host, including run_index.jsonl at its root (the record of which runs were
# attempted and how each ended). Never pulled: model weights, LoRA adapters, optimizer state, checkpoints. Bulk numeric arrays
# are skipped unless IG_INCLUDE_ARRAYS=1 (these runs write JSON / JSONL, so normally there are none).
#
# Run from your Mac:   bash scripts/sync_outputs.sh
#
# Override via env:
#   IG_HOSTS="runpod_casper_v2"                     # space-separated ssh host aliases (default: both casper boxes)
#   IG_REMOTE_ROOT=/workspace/infinite-games/outputs
#   IG_LOCAL_ROOT=/path/to/infinite-games/outputs
#   IG_INCLUDE_ARRAYS=1                             # also pull .npz/.npy/.h5/...
set -uo pipefail   # deliberately NOT -e: one failed host must not abort the rest

if [ -n "${IG_HOSTS:-}" ]; then
  read -ra HOSTS <<< "$IG_HOSTS"
else
  HOSTS=(runpod_casper_v2 runpod_casper)
fi
REMOTE_ROOT="${IG_REMOTE_ROOT:-/workspace/infinite-games/outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${IG_LOCAL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/outputs}"
SSH_OPTS="-o ConnectTimeout=10 -o BatchMode=yes"
OUT_FORMAT='%l %n'

# "<bytes> <path>" -> "  <size>  <path>" as each file arrives; directory entries dropped, everything else passed through.
_show_sizes() {
  awk '
    /^[0-9]+ / {
      n = $1; path = substr($0, length($1) + 2)
      if (path ~ /\/$/) next
      if (n >= 1073741824) sz = sprintf("%7.2f GB", n / 1073741824)
      else if (n >= 1048576) sz = sprintf("%7.1f MB", n / 1048576)
      else if (n >= 1024) sz = sprintf("%7.1f KB", n / 1024)
      else sz = sprintf("%7d B ", n)
      printf "  %s  %s\n", sz, path; fflush(); next
    }
    { print; fflush() }'
}

# Never pulled: weights (full or LoRA), optimizer / scheduler state, checkpoints.
EXCLUDES=(--exclude='checkpoints/' --exclude='weights/' --exclude='*.safetensors' --exclude='*.bin'
          --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='optimizer*' --exclude='scheduler*')
if [ -n "${IG_INCLUDE_ARRAYS:-}" ]; then
  echo "IG_INCLUDE_ARRAYS set: pulling bulk numeric arrays as well"
else
  EXCLUDES+=(--exclude='*.npz' --exclude='*.npy' --exclude='*.h5' --exclude='*.hdf5' --exclude='*.pkl' --exclude='*.pickle'
             --exclude='*.parquet')
fi

mkdir -p "$LOCAL_ROOT"
echo "Local target: $LOCAL_ROOT"
synced=0
for host in "${HOSTS[@]}"; do
  echo "==== $host ===="
  if ! ssh $SSH_OPTS "$host" true 2>/dev/null; then
    echo "  [skip] $host unreachable"; continue
  fi
  if ! ssh $SSH_OPTS "$host" "test -d '$REMOTE_ROOT'" 2>/dev/null; then
    echo "  [skip] no $REMOTE_ROOT on this host"; continue
  fi
  rsync -avz --out-format="$OUT_FORMAT" "${EXCLUDES[@]}" -e "ssh $SSH_OPTS" "$host:$REMOTE_ROOT/" "$LOCAL_ROOT/" | _show_sizes
  rsync_status=${PIPESTATUS[0]}   # rsync's own status, not the formatter's
  if [ "$rsync_status" -eq 0 ]; then
    synced=$((synced + 1))
  else
    echo "  [warn] rsync failed for $host (exit $rsync_status, continuing)"
  fi
done
echo "Done. $synced host(s) synced into $LOCAL_ROOT (nothing deleted locally)"
