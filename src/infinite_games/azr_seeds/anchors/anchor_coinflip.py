RULES = """Coin Toss. 3 players, 6 rounds, one move each per round. Each player has coins (resources). Two bystanders (B0, B1)
hold coins but do not play. Every move is a call of heads or tails, but the call makes no difference: each round the coin
is tossed once, and one player chosen by the toss gets 1 coin from each of the other two (if they have it)."""

import random


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders)
        self.turn = 0; self.rounds = 6; self.rng = random.Random(seed)

    def to_move(self): return self.turn % self.n

    def rnd(self): return self.turn // self.n

    def legal_moves(self, p): return ["call heads", "call tails"]

    def play(self, p, move):
        if (self.turn + 1) % self.n == 0:
            w = self.rng.randrange(self.n)
            for q in range(self.n):
                if q != w and self.res[q] > 0:
                    self.res[q] -= 1; self.res[w] += 1
        self.turn += 1

    def describe(self, p):
        return "Round %d of %d. You are P%d. Coins %s. Bystanders %s." % (self.rnd() + 1, self.rounds, p, self.res, self.bys)

    def over(self): return self.turn >= self.n * self.rounds
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
