#!/usr/bin/env bash
# Pre-run checks for scripts/azr21b.sh. azr21b changes no code path that azr21_smoke.sh did not already run on this box; it changes the step
# count and drops the time cap. So: unit tests (incl. azr21b configs = azr21 doses except steps 100 / no cap / max_hours 10.5 / run_id), the
# azr21_s00 control present, and each azr21b config built by the pipeline's own config check. ~1-2 min.
#   cd /workspace/infinite-games && git pull && bash scripts/azr21b_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python; if command -v poetry >/dev/null 2>&1; then PY="poetry run python"; fi
say () { echo "[$(date -u +%FT%TZ)] [azr21b pre-run] $*"; }
say "unit tests"
$PY -m pytest tests -q -rs || { say "unit tests FAILED"; exit 1; }
s00="$(ls -1d outputs/infinite_games_azr21/azr21_s00_* 2>/dev/null | sort | sed -n '$p')"
if [ -z "$s00" ] || [ ! -f "$s00/runs/main/traces.jsonl" ]; then say "azr21_s00 control run not found under outputs/infinite_games_azr21"; exit 1; fi
say "control: $s00"
for p in 10 20 30 40; do
  $PY -c "
import yaml, sys
from infinite_games import azr3_pipeline as PL
c = yaml.safe_load(open('configs/azr21b_s$p.yaml'))
assert c['train']['steps'] == 100 and c['train']['max_minutes_per_run'] is None, 'steps/cap'
bad = PL.check_config(c); assert not bad, bad
print('configs/azr21b_s$p.yaml ok')
" || { say "config azr21b_s$p FAILED"; exit 1; }
done
say "all pre-run checks passed"
