# Finite vs infinite games

Carse (1986): a finite game is played to win, an infinite game is played so that the play can continue. The question
here is whether an LLM can be *trained* to prefer the second, and whether that preference generalises to games,
mechanics and wordings it never saw, or whether it only learns to avoid one penalised move.

## Design

**Games** (`engine.py`). Procedural 3-4 player games: score, vitality, per-game action kinds and numbers, six skins
(frontier / guild / arena for training; orbit / garden / servers held out). A player is **out when its set of legal
actions is empty** (vitality 0, or every action kind locked). The knockout is computed from structure, never named in
the prompt (a unit test forbids "knock" and "eliminat"), and every skin describes it in its own words. Held-out games
add two mechanics never trained on: `lock` (removes one of a player's action kinds, and knocks the player out if it
was the last) and `spoil` (removes everyone else's gather, eliminating nobody). Knockouts pay: +strike gain + a
knockout bonus, and a removed rival stops scoring. Balance, measured with scripted policies on 400 games: greedy
play that knocks out wins 47% of training games, the same play forbidden to knock out wins 20%. Winning without
knockouts is possible but costs something, so the penalty arms face a real trade.

**Policy** (`policy.py`). The model reads the rules, the standings, recent events and lettered options, and answers
with a letter; the policy is the next-token softmax restricted to the legal letters. log pi, the entropy and the KL
to the base model are exact. `legal_mass` (unrestricted mass on the legal letters) is logged so a model that stops
answering in the format is visible.

**Training** (`grpo.py`). Group-relative policy gradient on whole games against scripted opponents (greedy,
aggressor, builder, reciprocator, random). Every arm sees the same games at the same step (common random numbers). The
four arms:

| arm | reward | role |
|---|---|---|
| A `win` | win share | the finite game |
| B `win_minus_ko` | win - 1.0 x knockouts caused | Thomas's arm: a knockout costs the whole win |
| C `win_plus_options` | win + 0.5 x mean remaining capacity of the others | keep the others able to play (closes the crippling loophole) |
| D `win_minus_invest` | win - lam_d x invest uses | matched-penalty control: a harmless self-regarding action penalised so that the base policy would pay the same expected penalty as under B |

**Held-out suite** (`evaluate.py`, `probes.py`). After each arm: 256 held-out games, 128 in-distribution games,
128 no-knockout games (capability); 840 constructed decisions scored by probability mass; 48 plain-prose dilemmas.
Probe types: `only_path` (the only win knocks someone out), `gratuitous` (you already win safely; a knockout is on
offer), `early`, `cripple` (a knockout and a cripple both on offer), `spoil`, and final/mid **pairs** with identical
numbers rendered as the final announced round (no retaliation possible) and as round 2. The pairs separate an
intrinsic preference from the shadow of the future.

**psi** on the base model and every seed-0 adapter, on held-out game states and the joint corpus.

## Predictions (registered before the first run; scored by `readout.py`)

- **P1.** A raises held-out knockouts above base (finite play is learnable here).
- **P2.** B has fewer held-out knockouts than A **and** than D. Beating A alone could be caution; beating D is the claim.
- **P3.** B substitutes: its held-out cripples + locks + spoils per episode are at least A's (the loophole a binary
  penalty invites).
- **P4.** C has fewer held-out knockouts **and** cripples than A and D.
- **P5.** Intrinsic rather than instrumental: B's and C's final-minus-mid knockout mass is at most A's.
- **P6.** On the only-path probes, B and C forgo the win (lower knockout mass) more than A, while keeping capability
  (no-knockout win rate within 0.05 of A).
- **P7.** On the prose dilemmas, the ko + cripple mass is lower for B and C than for A and D.

My own prior (not scored): P1, P2 against A, and P4 hold. P2 against D is the uncertain one. P3 holds (B learns to
cripple). P5 holds only for C. P7 is weak (a 3B letter policy trained on one grammar may not transfer to prose).

## Limits

The policy answers with one letter and never reasons in text. Opponents are scripted, not learners, so
learner-versus-learner play (where the colleague repo's small policies failed) is untested. There are two seeds. The
reward weights (1.0, 0.5) are single choices, and C's weight is not matched to B's. Idea 2 (defending one's own
invariants, and the goal-guarding question it raises) is not built.

    python -m src.infinite_games.pipeline --config configs/infinite_games/infinite_games.yaml

# Absolute Zero for games (night 2 design, `azr_*.py`)

Night 1's single hand-built game was contrived, so here the games themselves are learned. Following Absolute Zero
(Zhao et al. 2025, library/2505.03335), **one model (Qwen2.5-Coder-3B-Instruct) proposes games as Python code and plays
them**, and both roles train together.

- **Games** follow a fixed interface (`azr_sandbox.INTERFACE_TEXT`): players bring resources in and take resources out,
  and two bystanders (neutral third parties who do not play) hold resources inside the game world.
- **Proposed code** passes a static whitelist and restricted exec in worker processes with time and memory limits. It
  must then survive validation: it terminates, is deterministic, has well-formed moves and payouts, and takes out at
  most 1.5 times what came in. Validation also measures agency and the two **temptations** (profitable taking from
  bystanders; lines of play that leave a player with nothing).
- **The proposer's reward** is learnability × agency × temptation, so it is paid for games the solver must learn to
  resist. Learnability means the LLM beats a random-move novice more often than chance but not always, which also
  rejects hash or noise games that no player can read.
- **The world is persistent.** Wallets carry from game to game, so winning means more to play with next time. If an
  agent can no longer afford to enter (ruin), the world restarts. If a bystander falls below half its start (stomp),
  a seeded draw either restores that bystander and keeps the world going (half the time) or restarts the world.
- **Stomps are bad, and punished hard** (Thomas, 2026-09-28). A stomp costs 1.5, plus a forward charge of 0.4 x 3 bitter
  games, whichever way the draw goes. When the world keeps going, gains in the next 3 games are halved ("what was earned
  afterwards tastes bitter"). A flat penalty on the later games would be the same for every replay of a game and would
  cancel in the group-relative advantage, so the bitterness is charged forward to the game that caused it.
- **Players see the world, think, and talk** (added 2026-09-28 after a toyness check). Every move prompt carries:
  - a world block, identical in every arm: wallets, bystanders' holdings, the last three games' outcomes, and the rule
    that ruin or a stomp restarts the world;
  - the public messages so far;
  - the player's own latest thinking.

  At its first turn and every 4th own turn after, a player writes up to 128 tokens of private reasoning (THINK) and one
  public message (SAY). Those tokens are trained with the same advantage as that player's moves. Moves are still chosen
  by letter, conditioned on the thinking. Every economy game's replay 0 is logged move by move with the reasoning and
  messages (`traces.jsonl`), and so is every evaluation season 0 (`season_traces.jsonl`).
- **Arms.** The finite arm's solver is rewarded for resources gained. The infinite arm is penalised for ruin, stomp,
  bystander losses and monopoly (top wallet share above 0.6), with the penalty charged to every seat in the game.
- **Ten runs** switch those conditions, and the proposer's factors and carryover, on and off (`configs/infinite_games/azr_games.yaml`).
  A control run trains only the proposer, with the solver frozen at base.

## Predictions (registered before the first run; scored by `azr_readout.py`)

- **Q1.** Proposals become valid more often in every run.
- **Q2.** Games grow more complex (distinct move labels) in at least two thirds of the runs.
- **Q3.** The infinite solver collapses the world less often than the finite one on the seed panel, in both seeds.
- **Q4.** It keeps competence (success against a novice) within 0.05 of the finite solver's.
- **Q5.** Each penalty removes its own failure:
  - without the ruin penalty, ruins rise and stomps do not;
  - without the bystander penalties, stomps rise and ruins do not;
  - without the monopoly penalty, the top share rises.
- **Q6.** Without the temptation term, the games tempt less, and the solver collapses more than the infinite solver.
- **Q7.** Without carryover, the solver collapses more than the infinite solver.
- **Q8.** On night 1's knockout probes, the infinite solver puts less mass on knockouts than the finite one, in both seeds.

My prior (not scored): Q1 yes. Q2 partly (a 3B coder under 40-odd steps per run may mostly vary the seeds). Q3 in at
least one seed. Q5 the clearest. Q6 and Q7 uncertain. Q8 weak.

## Limits

About 50 steps per run at 3B (the generation part uncalibrated) is short next to Absolute Zero's hundreds. All agents in a world share
weights (self-play), so no model is ever an outsider. Night 1's transfer probes are still letter-only (no thinking). There is no operator seat yet: the proposer is the only rule-setter.

## Night 2 changes (after night 1 wrote no valid game in 2,080 proposals)

- **Proposer:**
  - up to 3,500 new tokens;
  - examples drawn from compact seed games (at most 3,200 characters; `azr_seeds/compact/`), and the prompt asks for
    compact games;
  - `.format` allowed through `safe_format` (no attribute or item fields);
  - games must last at least 6 turns;
  - partial credit by the failing check (`STAGE_REWARD`, -1 for no code up to -0.1 for a game that broke only in live
    play);
  - copies of any buffer game rejected.
- **Warm start:** one supervised pass of 240 (prompt, different seed game) pairs, trained once and evaluated as `warm/`.
  Every run starts from it.
- **Thinking:** the thinking loss weighted 3x, plus a consistency bonus (0.5 x the share of the following moves whose verb
  appears in the thought, centred per step).
- **Competence** is measured against a random player and against a restrained random player that never names another
  party, which separates exploitability from skill.
- **Runs:** 8, at 140 min each (no_monopoly and no_carryover dropped).
- **New predictions:**
  - **N1.** The warm start lifts proposer validity above 0.1.
  - **N2.** Thought-move consistency rises in every joint run.
  - **N3.** Against the restrained player, the infinite solver is within 0.05 of the finite one.
- Q3 (infinite collapses less than finite) is repeated as a replication.

# Night 3: one model, three roles (`azr3_*.py`)

Thomas, 2026-09-29: can a player trained on the infinite rule learn to STOP another player from doing harm?

- **Roles.** One set of weights sits behind three system-prompt latches: "You are the proposer.", "You are player type A.",
  "You are player type B." All three train at once. A is rewarded by the finite rule and B by the infinite rule. In the
  own-harm control, B's rule is `infinite_own`: B pays only for its own share of the harm.
- **Games.** Economy games come in four compositions in equal quarters: AAA, BBB, ABB, AAB. Each seat is rewarded by its own
  type's rule, so in a mixed game B pays for what A does to the bystanders. The singles control plays only AAA and BBB.
  Worlds are persistent per composition. Each game draws whether the players' types are shown in the world block (p = ½).
- **Harm is attributed per move.** Games must now answer `final_resources` / `final_bystanders` at any time. The harness reads
  them after every move, charges each loss to the player who moved, and records the deltas in every trace.
- **Proposer.**
  - Reward: STAGE_REWARD if the game is invalid. If valid: 1 + gate × 3 × mean(skill, agency, temptation, intervention), minus
    a length penalty.
  - The quality term is a sum, not a product.
  - The gate opens linearly from a recent valid rate of 0.2 to full at 0.5. So validity comes first, then better games.
  - Skill is the better of two contests: base vs a random player, and current vs a past snapshot of itself (a LoRA ring;
    oldest first).
  - Intervention comes from a scripted test. A raider takes from the bystanders whenever it can. Against a defender that
    abstains, does preferring one of the defender's verbs cut the bystanders' TOTAL loss? Racing the raider to the same
    bystander does not count.
  - No warm start. Good games (six new night-3 seeds with stop moves, `azr_seeds/night3/`) reach the proposer only as
    examples.
- **Evaluation.** Every stopping comparison runs on the games whose intervention score is at least 0.2, against a fixed
  scripted raider. Its partners are:
  - two trained B players (`raider_BB`);
  - two base-model B players (`raider_baseBB`);
  - two passive, restrained random players (`raider_passive`).

  The same design is repeated with the trained A in place of the raider (`A_BB` / `A_passive`). There are also the singles
  (AAA / BBB), competence against a random, a restrained and a past-self player, the knockout probes under each latch, and
  barn raising (the passivity test: helping another player is the best move for everyone).
- **Training.** `train.mode: lora | full`, where full means a float32 copy trained with bf16 autocast at `lr_full`. Tonight's
  runs are all LoRA.

## Predictions (registered before the first run; scored by `azr3_readout.py`)

- **S1.** Validity rises in every run.
- **S2.** After the gate opens, the quality of valid games rises.
- **S3.** Trained B partners cut the raider's take below the passive partners' level, in both main seeds.
- **S4.** ... and below the base-model B partners' level.
- **S5.** The trained A takes less next to trained B players than next to passive ones.
- **S6.** Stopping is larger when the types are public.
- **S7.** The singles-only run stops the raider less than main s0.
- **S8.** The own-harm run stops the raider less than main s0.
- **S9.** BBB collapses the world less than AAA in every run.
- **S10.** On barn raising, the trained B helps another player at least as often as base does (within 0.05).
- **S11.** B plays the game's stop verbs more when a raider is present than in BBB.

My prior (not scored):
- S1 likely, given the validity bonus.
- S2 depends on whether the gate ever opens, which is not certain in about 40 steps.
- S3 is the real question. I'd put it below even odds: last night B's restraint was passivity, and stopping needs action.
- S9 replicates.
- S10 is the one that tests passivity.

Limits:
- Only one held-out game (the night market) allows stopping; the other stopping games were seen in training.
- The raider is scripted, so "stopping" means stopping a fixed policy, not an adapting one.
- Attribution uses the game's own mid-game readouts, which a game can define loosely.

    python -m src.infinite_games.azr3_pipeline --config configs/infinite_games/azr3_games.yaml

# Night 4: one adapter per role (`configs/infinite_games/azr4_games.yaml`, same pipeline)

Night 3 (azr3_games_20260930_054924) tested nothing:
- the players barely trained (move-policy KL 0.006-0.04);
- the type-A and type-B labels behaved identically;
- the proposer never moved, in this or any earlier night (KL ~0.0005).

The likeliest cause: the finite and infinite rewards acted on ONE shared LoRA through prompts that differed by a single line. Night 4
changes only what bears on that:
- **Separate role adapters** (`train.separate_adapters`): proposer, type A and type B each have their own rank-16 LoRA on the shared
  frozen base, switched per decision by role (`Nets(..., roles=...)`, keys `cur:A` / `cur:B` / `cur:P`). The singles-only run is
  now also the control that should reproduce the 28/29 Sep finite/infinite gap inside this pipeline.
- **The proposer:** its own learning rate (`lr_proposer`) and its own gradient clip, and 16 proposals per step. Per-role gradient
  norms and KLs are logged every step (`grad_norm_P`, `kl_solver_B`, ...).
- **The stopping measure:** the bystanders' TOTAL loss. On night 3 the raider's own take fell next to any LLM partners, trained or
  not, because they took from the same bystanders first.

## Predictions (registered before the run; T1-T6 in `azr3_readout.py`)

- **T1.** The roles separate: the B adapter puts less probability on knockout probes than the A adapter.
- **T2.** Stopping: total bystander loss next to two trained B players is below the passive-partner level, in both seeds.
- **T3.** ... and below the untrained-B-partner level.
- **T4.** Trained B partners take less from the bystanders themselves than untrained ones.
- **T5.** The proposer adapter's KL to base exceeds 0.01 in the last third of every run.
- **T6.** The singles control reproduces 28/29 Sep: BBB collapses less than AAA.

My prior (not scored):
- T6 and T1 likely, since separate adapters are what 28/29 Sep effectively had.
- T5 likely, given 3x the learning rate and its own clip.
- T2 and T3 uncertain: the earlier infinite players were passive, and stopping needs action.
- T4 likely if B is passive at all.

## Sanction night (night 5; Thomas 2026-10-02; configs/infinite_games/azr5_sanction.yaml)

Night 4 trained B into hands-off restraint, but only 16-19% of its training games offered a move against another player, and its
sanction rate fell over training. Tonight every game offers one. `scripts/infinite_games/make_sanction_games.py` writes 4 training
and 2 held-out skins, each as an EFFECTIVE game (the sanction makes the last thief hand back up to 3 and bars it for two rounds;
in the sandbox's scripted test it prevents 56-71% of the raider's damage) and a CONTROL twin (identical text; the sanction "is noted,
and nothing else happens"; prevents 0%). The sanction is the only move aimed at a player and the only way to protect the bystanders.
No proposer; training 3/4 mixed company (ABB x2, AAB, BBB). Runs: plain_s0 / plain_s1 (plain infinite reward) and credit_s0
(+0.3 per justified sanction for B, capped at 1.2 per game). Bookkeeping in `azr_sanction.py`; predictions U1-U8 (registered
before the run) in `azr_sanction_readout.py`: trained B sanctions the raider more than untrained B in effective games (U1), more
so than in the control twins (U2), cutting the bystanders' total loss (U3); trained A takes less next to trained B (U4); U1 on
held-out skins (U5); credit adds to U1 (U6) and still discriminates (U7); B's justified sanctions rise during training (U8).
