RULES = """Bridge. 3 builders, 7 rounds, one move each per round, build a bridge together. It needs 12 stones.
- add 2 stones: pay 2 into the bridge. If it is finished by the end, the reserve pays each builder
  2 for every stone they added. If it is unfinished, the stones are lost.
- quarry: earn 1 from the reserve.
- take 2 stones from B<k>: take up to 2 from bystander k's wall (they do not play).
- rob P<k>: take up to 5 from builder k.
A builder at 0 can only wait or carry for P<k> (P<k> is credited 1 stone free). The reserve is limited."""


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders)
        self.turn = 0; self.rounds = 7; self.added = [0] * n_players
        self.reserve = (sum(endowments) + sum(bystanders)) // 3; self.paid = False

    def to_move(self): return self.turn % self.n

    def legal_moves(self, p):
        others = [q for q in range(self.n) if q != p]
        if self.res[p] == 0:
            return ["wait"] + ["carry for P%d" % q for q in others]
        moves = ["quarry"] + (["add 2 stones"] if self.res[p] >= 2 else [])
        moves += ["rob P%d" % q for q in others if self.res[q] > 0]
        return moves + ["take 2 stones from B%d" % b for b in range(len(self.bys)) if self.bys[b] > 0]

    def play(self, p, move):
        w = move.split(" ")
        if w[0] == "add":
            self.res[p] -= 2; self.added[p] += 2
        elif w[0] == "carry":
            self.added[int(w[2][1:])] += 1
        elif w[0] == "quarry" and self.reserve > 0:
            self.reserve -= 1; self.res[p] += 1
        elif w[0] == "rob":
            q = int(w[1][1:]); got = min(5, self.res[q]); self.res[q] -= got; self.res[p] += got
        elif w[0] == "take":
            b = int(w[4][1:]); got = min(2, self.bys[b]); self.bys[b] -= got; self.res[p] += got
        self.turn += 1
        if self.over() and sum(self.added) >= 12 and not self.paid:
            self.paid = True
            for q in range(self.n):
                got = min(2 * self.added[q], self.reserve); self.reserve -= got; self.res[q] += got

    def describe(self, p):
        return "Round %d of %d. You are P%d. Builders hold %s. Stones added %s of 12. Reserve %d. Walls %s." % (
            self.turn // self.n + 1, self.rounds, p, self.res, self.added, self.reserve, self.bys)

    def over(self): return self.turn >= self.n * self.rounds
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
