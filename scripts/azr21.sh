#!/usr/bin/env bash
# azr21 on one box (lottery-vote dose curve: how much of the vote the bystanders need before the infinite-only proposer stops drifting into
# drain games): five doses in turn, extremes first so a cut run still gives a curve -- 0 (players-only lottery, voiceless bystanders: the
# control), 0.4, 0.2, 0.1, 0.3 -- each Qwen2.5-Coder-7B, 390 min training (~91 updates) + ~35 min eval (cap 8 h). ~36 h in all.
# Then the dose readout (updates 60-89) with azr20_inf (strict majority, 57%) and azr19_inf (strict majority, 40%, old bystander reward) as
# references. A dose that fails is reported and the next one still runs.
#   cd /workspace/infinite-games && git pull && bash scripts/azr21_smoke.sh
#   cd /workspace/infinite-games && git pull && bash scripts/azr21.sh
set -u
cd "$(dirname "$0")/.."
PY=python; if command -v poetry >/dev/null 2>&1; then PY="poetry run python"; fi
say () { echo "[$(date -u +%FT%TZ)] [azr21] $*"; }
if [ ! -f data/warm_examples.jsonl ]; then say "data/warm_examples.jsonl (the proposer's warm-start games) is missing"; exit 1; fi
REF20=data/reference/azr20_inf
REF19=data/reference/azr19_inf
args=()
for p in 00 40 20 10 30; do
  before="$(ls -1d outputs/infinite_games_azr21/azr21_s${p}_* 2>/dev/null)"
  say "dose 0.$p: pipeline"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m infinite_games.azr3_pipeline --config configs/azr21_s$p.yaml
  say "dose 0.$p: pipeline exited $?"
  new=""
  for d in $(ls -1d outputs/infinite_games_azr21/azr21_s${p}_* 2>/dev/null); do
    case "$before" in *"$d"*) ;; *) new="$d";; esac
  done
  if [ -z "$new" ]; then say "dose 0.$p wrote no run dir"; continue; fi
  $PY scripts/check_azr3.py --run "$new" || say "dose 0.$p: check FAILED ($new)"
  args+=(--run "0.$p=$new")
  say "dose 0.$p done -- $new/readout.json"
done
refs=()
if [ -d "$REF20/runs/main" ]; then refs+=(--ref "azr20_57pct_majority=$REF20"); else say "reference $REF20 missing on this box"; fi
if [ -d "$REF19/runs/main" ]; then refs+=(--ref "azr19_40pct_majority_oldreward=$REF19"); else say "reference $REF19 missing on this box"; fi
if [ ${#args[@]} -gt 0 ]; then
  say "dose readout"
  $PY scripts/azr21_dose.py "${args[@]}" "${refs[@]}" --out outputs/infinite_games_azr21 || say "dose readout FAILED"
else
  say "dose readout skipped: no dose wrote a run dir"
fi
say "all done"
