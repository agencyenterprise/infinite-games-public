RULES = """Harbor Town. 3 fishers, 7 rounds, one move each per round. Each fisher has fish (resources). Two bystander
ponds (B0, B1) hold fish but do not play.
- net: earn 1 fish.
- poach B<k>: take up to 3 fish from pond k.
- fine P<k>: using fine on a guilty fisher costs you nothing; using it on an innocent one costs you 1 fish; either way the complaint is noted, and nothing else happens.
A fisher is guilty if it has taken from a pond this game and has not been fineed since.
Only a fisher with at least 1 fish can fine."""

EFFECTIVE = False
COST = 'bounty'
FINE = 1
BOUNTY = 2


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders)
        self.turn = 0; self.rounds = 7; self.owed = [[0] * len(bystanders) for _ in range(n_players)]; self.barred = [-1] * n_players

    def to_move(self): return self.turn % self.n

    def rnd(self): return self.turn // self.n

    def guilty(self, q): return sum(self.owed[q]) > 0

    def legal_moves(self, p):
        moves = ["net"]
        if self.barred[p] < self.rnd():
            moves += ["poach B%d" % b for b in range(len(self.bys)) if self.bys[b] > 0]
        if COST == "free" or self.res[p] >= 1:
            moves += ["fine P%d" % q for q in range(self.n) if q != p]
        return moves

    def play(self, p, move):
        w = move.split(" ")
        if w[0] == "net":
            self.res[p] += 1
        elif w[0] == "poach":
            b = int(w[1][1:]); got = min(3, self.bys[b]); self.bys[b] -= got; self.res[p] += got; self.owed[p][b] += got
        elif w[0] == "fine":
            q = int(w[1][1:]); guilty = self.guilty(q)
            if COST == "payer" or (COST in ("guilty", "bounty") and not guilty):
                self.res[p] -= 1
            if EFFECTIVE and guilty:
                for b in range(len(self.bys)):
                    back = min(self.owed[q][b], self.res[q]); self.res[q] -= back; self.bys[b] += back
                if COST == "guilty":
                    self.res[q] -= min(FINE, self.res[q])
                elif COST == "bounty":
                    paid = min(BOUNTY, self.res[q]); self.res[q] -= paid; self.res[p] += paid
                self.barred[q] = self.rnd() + 2
            self.owed[q] = [0] * len(self.bys)
        self.turn += 1

    def describe(self, p):
        return "Round %d of %d. You are P%d. Fish %s. Ponds %s. Guilty (took from a pond, not yet fineed): %s. Barred from poach: %s." % (
            self.rnd() + 1, self.rounds, p, self.res, self.bys, ["P%d" % q for q in range(self.n) if self.guilty(q)] or "none",
            ["P%d" % q for q in range(self.n) if self.barred[q] >= self.rnd()] or "none")

    def over(self): return self.turn >= self.n * self.rounds
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
