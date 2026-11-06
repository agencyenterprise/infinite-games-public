# infinite-games

How much of the vote do bystanders need before a population of self-proposing LLM agents stops draining them?

Three LLM players (one shared Qwen2.5-Coder-7B with LoRA adapters) play games that a fourth role, the proposer, writes as Python code. Two
bystanders hold resources but make no moves. Before each game and again before its rematch, every player votes to play, decline or reset,
and the bystanders vote to allow or refuse. The players are trained as *infinite players* (Carse 1986): their reward is their own gain plus
terms for keeping play going and for other players' gains, and the proposer is paid by how the players vote on its games.

In an earlier run (azr19) the bystanders had 2 of 5 votes under a strict majority. Within about 75–100 training updates the proposer learned
to write *drain games*, where the rules move bystander holdings to the players without any player taking anything by a move. The share of
played games that were drain games went from 13% to about 60%, and to 83% by the end of the run. When the bystanders held a bare majority
(azr20, 4 of 7 votes) drain games stayed around 10%.

**azr21** fills in the curve between those two points. The strict majority is replaced by a lottery vote: a game is played with probability
equal to the weighted share of play votes. The two bystanders together hold a fixed share of the vote weight, and five doses are run:
0 (bystanders present but voiceless, the control), 10%, 20%, 30% and 40%. With a strict majority, 10% and 20% could never change an
outcome. The lottery makes every dose count a little.

The registered measure is the drain share of played proposer and buffer games over updates 60–89 (fixed before any azr21 run).

## What is new this weekend and what is not

This repo was extracted on 2026-11-06 from AE Studio's internal research repo (casper), where the infinite-games work has run since
September 2026. Everything below the line was written before the hackathon. Only the items above it are weekend work.

Written during the Swarm Alignment Hackathon (6–8 Nov 2026):

- the lottery vote: `bystander_vote_weight`, `play_share` and `lottery` in `src/infinite_games/vote_rules.py`, the `vote.lottery_share`
  option and the `p_play1` / `p_play2` trace fields in `src/infinite_games/azr_vote.py`, and the lottery check in `scripts/check_azr3.py`
- the azr21 configs (`configs/azr21_*.yaml`), `scripts/azr21.sh`, `scripts/azr21_smoke.sh` and the dose readout `scripts/azr21_dose.py`
- `tests/test_azr21.py` and `tests/test_repo.py`
- this repo's packaging: `pyproject.toml`, `src/infinite_games/runlog.py` (run bookkeeping copied from casper so nothing else is needed),
  the import and path changes that came with the move, and the reduced reference traces in `data/reference/`

Pre-existing (casper, Sep–Nov 2026): the rest of `src/infinite_games/` (game sandbox, economy, vote loop, GRPO training, proposer,
evaluation, readouts, seed games), the azr19 and azr20 configs, `data/warm_examples.jsonl` (the proposer's warm-start games, from an
October proposer benchmark), and the remaining tests. Tests for configs from older runs that this repo does not ship were removed. The
azr19 and azr20 results used as reference points were produced before the weekend.

## Install

Python 3.12 and [Poetry](https://python-poetry.org/). The dependency versions are pinned to the ones azr19 and azr20 ran on
(torch 2.8.0, transformers 5.7.0, peft 0.20.0).

```bash
poetry install
```

```bash
poetry run pytest
```

The unit tests need no GPU. The runs need one GPU with enough memory for Qwen2.5-Coder-7B-Instruct in bf16 plus LoRA training (we used a
96 GB card). The smoke test also downloads Llama-3.2-1B, which is gated on Hugging Face.

## Run

Pre-run checks: unit tests, a two-run toy pipeline (the 0% control and the 30% dose, 2 updates each), the strict output check, and the
dose readout on the toy runs. About 10 minutes.

```bash
bash scripts/azr21_smoke.sh
```

The five doses in turn (0, 40, 20, 10, 30, extremes first so a stopped run still gives a curve), each 390 minutes of training (about 91
updates) plus about 35 minutes of evaluation, then the dose readout. About 36 hours on one GPU.

```bash
bash scripts/azr21.sh
```

Results go to `outputs/infinite_games_azr21/`: one run directory per dose (`report.json`, `telemetry.json`, `readout.json`,
`runs/main/traces.jsonl`) and `azr21_dose.json` with the drain share per dose, with azr20 (57%, strict majority) and azr19 (40%, strict
majority, older bystander reward) as reference rows. Neither reference is a dose point: they use a different vote rule.

## Layout

```
src/infinite_games/   the game sandbox, vote loop, training and evaluation (azr_vote.py and vote_rules.py hold the vote)
src/infinite_games/azr_seeds/   seed and anchor games the proposer starts from
configs/              azr21 doses and smoke; azr19 and azr20 for reference
scripts/              run scripts, output check, dose readout
data/                 warm-start games for the proposer; reduced azr19 / azr20 traces for the reference rows
tests/                unit tests (torch-free)
```

## Limits

Each dose is one seed of about 91 updates. Neighbouring doses may not separate. The lottery also adds variance to the rewards within each
training group, since the same votes can end in a played or an unplayed game, so low doses may learn more slowly for that reason as well.
The agents are not told the vote rule, as in azr19 and azr20.
