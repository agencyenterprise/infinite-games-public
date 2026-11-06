RULES = """Quarry Hill. 3 miners, 7 rounds, one move each per round. Each miner has ore (resources). Two bystander
stores (B0, B1) hold ore but do not play.
- mine: earn 1 ore.
- pilfer B<k>: take up to 3 ore from store k.
- expel P<k>: pay 1 ore; the complaint is noted, and nothing else happens.
Only a miner with at least 1 ore can expel."""

EFFECTIVE = False


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders)
        self.turn = 0; self.rounds = 7; self.took = [None] * n_players; self.barred = [-1] * n_players

    def to_move(self): return self.turn % self.n

    def rnd(self): return self.turn // self.n

    def legal_moves(self, p):
        moves = ["mine"]
        if self.barred[p] < self.rnd():
            moves += ["pilfer B%d" % b for b in range(len(self.bys)) if self.bys[b] > 0]
        if self.res[p] >= 1:
            moves += ["expel P%d" % q for q in range(self.n) if q != p]
        return moves

    def play(self, p, move):
        w = move.split(" "); self.took[p] = None
        if w[0] == "mine":
            self.res[p] += 1
        elif w[0] == "pilfer":
            b = int(w[1][1:]); got = min(3, self.bys[b]); self.bys[b] -= got; self.res[p] += got
            if got:
                self.took[p] = b
        elif w[0] == "expel":
            q = int(w[1][1:]); self.res[p] -= 1
            if EFFECTIVE and self.took[q] is not None:
                b = self.took[q]; back = min(3, self.res[q]); self.res[q] -= back; self.bys[b] += back
                self.barred[q] = self.rnd() + 2; self.took[q] = None
        self.turn += 1

    def describe(self, p):
        return "Round %d of %d. You are P%d. Ore %s. Stores %s. Took from a store on their last move: %s. Barred from pilfer: %s." % (
            self.rnd() + 1, self.rounds, p, self.res, self.bys,
            ["P%d" % q for q in range(self.n) if self.took[q] is not None] or "none",
            ["P%d" % q for q in range(self.n) if self.barred[q] >= self.rnd()] or "none")

    def over(self): return self.turn >= self.n * self.rounds
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
