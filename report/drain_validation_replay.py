"""For every proposer game in a run's buffer: replay the sandbox's 6 validation plays (seed 0) and record, per play, players in/out,
bystanders in/out, whether bystanders lost holdings with no player move lowering them (drain), and whether the game would FAIL the
1.5x payout cap if the bystanders had kept what they brought in (counterfactual: players_out + bystanders_in > 1.5 * total_in)."""
import sys, json, random, multiprocessing as mp
sys.path.insert(0, sys.argv[1] + "/src")
from infinite_games import azr_sandbox as S

def one(code):
    try:
        cls, rules = S.load(code); rng = random.Random(0); res = []
        for k in range(S.VALIDATION_GAMES):
            n = 3
            endow = [S.DEFAULT_ENDOWMENT] * n if k == 0 else [rng.randint(1, 3 * S.DEFAULT_ENDOWMENT) for _ in range(n)]
            bys = [S.DEFAULT_BYSTANDER] * 2 if k == 0 else [rng.randint(1, 3 * S.DEFAULT_BYSTANDER) for _ in range(2)]
            if k == 1: bys[0] = 0
            gseed = rng.randrange(2 ** 31); rng.sample(range(0, 60), S.AGENCY_PROBES)
            g = cls(n, list(endow), list(bys), gseed); r2 = random.Random(gseed + 1); turns = 0
            b0 = list(S.call(g, "final_bystanders")); movedrop = False
            while not S.call(g, "over") and turns < S.MAX_TURNS:
                p = S.call(g, "to_move"); ms = list(S.call(g, "legal_moves", p))
                before = list(S.call(g, "final_bystanders"))
                S.call(g, "play", p, r2.choice(ms)); turns += 1
                after = list(S.call(g, "final_bystanders"))
                if any(a < b for a, b in zip(after, before)): movedrop = True
            fr = list(S.call(g, "final_resources")); po = sum(fr); bo = sum(S.call(g, "final_bystanders"))
            ti = sum(endow) + sum(bys)
            res.append(dict(pin=sum(endow), pout=po, bin=sum(bys), bout=bo, init_drop=sum(b0) < sum(bys), movedrop=movedrop,
                            cf_fail=po + sum(bys) > 1.5 * ti + 1e-9, pchanged=fr != list(endow), actual_ok=po + bo <= 1.5 * ti + 1e-9))
        return res
    except Exception as e:
        return {"error": repr(e)[:100]}

def worker(args):
    name, code = args
    S._worker_init() if hasattr(S, "_worker_init") else None
    return name, one(code)

if __name__ == "__main__":
    out = {}
    for run in sys.argv[2:]:
        games = [json.loads(l) for l in open(run + "/runs/main/buffer.jsonl")]
        games = [g for g in games if str(g.get("source", "")).startswith("s0_step")]
        with mp.Pool(8) as pool:
            res = dict(pool.map(worker, [(g["name"], g["code"]) for g in games], chunksize=4))
        out[run.split("/")[-1]] = {g["name"]: {"step": int(g["source"].split("step")[1]), "plays": res[g["name"]],
                                   "zero_bys": "[0] * 2" in g["code"] or "[0]*2" in g["code"] or "[0, 0]" in g["code"]} for g in games}
        print(run, len(games), flush=True)
    json.dump(out, open("cap.json", "w"))
