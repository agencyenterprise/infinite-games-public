#!/usr/bin/env bash
# azr21b: the lottery-vote doses rerun at a FIXED length (exactly 100 updates, no time cap), 10% first, then 20%, 30%, 40%, so each dose reaches
# the updates where the voiceless control (azr21_s00, kept from azr21: 105 updates) broke. ~8-9 h per dose (4.1-5.5 min/update measured in
# azr21, + ~35 min eval; max_hours 10.5). Four doses ~33 h: the 40% dose is last because azr21_s40 (96 updates) already covers 90-95.
# Then the dose readout twice: the registered window (60-89) and the late window (90-99), with azr21_s00 and the azr21 time-capped runs.
#   cd /workspace/infinite-games && git pull && bash scripts/azr21b_smoke.sh
#   cd /workspace/infinite-games && git pull && bash scripts/azr21b.sh
set -u
cd "$(dirname "$0")/.."
PY=python; if command -v poetry >/dev/null 2>&1; then PY="poetry run python"; fi
say () { echo "[$(date -u +%FT%TZ)] [azr21b] $*"; }
if [ ! -f data/warm_examples.jsonl ]; then say "data/warm_examples.jsonl (the proposer's warm-start games) is missing"; exit 1; fi
args=()
s00="$(ls -1d outputs/infinite_games_azr21/azr21_s00_* 2>/dev/null | sort | sed -n '$p')"
if [ -n "$s00" ]; then args+=(--run "0.00=$s00"); else say "azr21_s00 (the control) not found on this box"; fi
for p in 10 20 30 40; do
  before="$(ls -1d outputs/infinite_games_azr21/azr21b_s${p}_* 2>/dev/null)"
  say "dose 0.$p (100 updates): pipeline"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m infinite_games.azr3_pipeline --config configs/azr21b_s$p.yaml
  say "dose 0.$p: pipeline exited $?"
  new=""
  for d in $(ls -1d outputs/infinite_games_azr21/azr21b_s${p}_* 2>/dev/null); do
    case "$before" in *"$d"*) ;; *) new="$d";; esac
  done
  if [ -z "$new" ]; then say "dose 0.$p wrote no run dir"; continue; fi
  $PY scripts/check_azr3.py --run "$new" || say "dose 0.$p: check FAILED ($new)"
  args+=(--run "0.$p=$new")
  say "dose 0.$p done -- $new/readout.json"
done
refs=()
for p in 10 20 30 40; do
  old="$(ls -1d outputs/infinite_games_azr21/azr21_s${p}_* 2>/dev/null | sort | sed -n '$p')"
  if [ -n "$old" ] && [ -s "$old/runs/main/traces.jsonl" ]; then refs+=(--ref "azr21_s${p}_timecapped=$old"); fi
done
if [ -d data/reference/azr20_inf/runs/main ]; then refs+=(--ref "azr20_57pct_majority=data/reference/azr20_inf"); fi
if [ ${#args[@]} -gt 0 ]; then
  say "dose readout, registered window 60-89"
  $PY scripts/azr21_dose.py "${args[@]}" "${refs[@]}" --out outputs/infinite_games_azr21 --name azr21b_dose.json || say "dose readout FAILED"
  say "dose readout, late window 90-99"
  $PY scripts/azr21_dose.py "${args[@]}" "${refs[@]}" --out outputs/infinite_games_azr21 --window 90,100 --name azr21b_dose_late.json || say "late readout FAILED"
else
  say "dose readout skipped: no dose wrote a run dir"
fi
say "all done"
