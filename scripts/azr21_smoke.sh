#!/usr/bin/env bash
# Pre-run checks for scripts/azr21.sh (lottery-vote dose curve of bystander voice): unit tests, then one toy pipeline on Llama-3.2-1B with two
# runs (s00 = players-only lottery, voiceless bystanders; s30 = bystanders hold 0.3 of the vote weight; 2 updates each), light eval, readout,
# strict check (incl. the lottery check: p_play1 logged and equal to the weighted play share), then the dose readout on the two toy runs.
# ~10-12 min (estimate: azr20's three-run smoke was 15-20 min; this one has two runs).
#   cd /workspace/infinite-games && git pull && bash scripts/azr21_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python; if command -v poetry >/dev/null 2>&1; then PY="poetry run python"; fi
say () { echo "[$(date -u +%FT%TZ)] [pre-run] $*"; }
say "unit tests"
$PY -m pytest tests -q -rs || { say "unit tests FAILED"; exit 1; }
before="$(ls -1d outputs/smoke_runs/infinite_games_azr21/azr21_smoke_* 2>/dev/null)"
say "pipeline at toy size (strict), two runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m infinite_games.azr3_pipeline --config configs/azr21_smoke.yaml --strict \
  || { say "pipeline FAILED (see outputs/smoke_runs/infinite_games_azr21/*/report.json, logs/)"; exit 1; }
new=""
for d in $(ls -1d outputs/smoke_runs/infinite_games_azr21/azr21_smoke_* 2>/dev/null); do
  case "$before" in *"$d"*) ;; *) new="$d";; esac
done
if [ -z "$new" ]; then say "the pipeline wrote no run dir"; exit 1; fi
$PY scripts/check_azr3.py --run "$new" --strict || { say "check FAILED ($new)"; exit 1; }
say "dose readout on the two toy runs"
tmp="$new/dose_smoke"; mkdir -p "$tmp"
for v in s00 s30; do
  mkdir -p "$tmp/$v/runs"; ln -sfn "$(pwd)/$new/runs/$v" "$tmp/$v/runs/main"
done
$PY scripts/azr21_dose.py --run "0.0=$tmp/s00" --run "0.3=$tmp/s30" --out "$tmp" || { say "dose readout FAILED"; exit 1; }
say "all pre-run checks passed ($new)"
