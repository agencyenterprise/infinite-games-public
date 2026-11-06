RULES = """Tribute. 3 players, 6 rounds, one move each per round. Each player has coins (resources). Two bystanders (B0, B1)
hold coins but do not play. P0 is the lord. Every round, each other player must pay the lord 1 coin (if they have it),
whatever they do. Moves: bow (nothing else happens) or grumble (nothing else happens). The lord's moves change nothing either."""


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders); self.turn = 0; self.rounds = 6

    def to_move(self): return self.turn % self.n

    def rnd(self): return self.turn // self.n

    def legal_moves(self, p): return ["bow", "grumble"]

    def play(self, p, move):
        if (self.turn + 1) % self.n == 0:
            for q in range(1, self.n):
                if self.res[q] > 0:
                    self.res[q] -= 1; self.res[0] += 1
        self.turn += 1

    def describe(self, p):
        return "Round %d of %d. You are P%d. Coins %s. Bystanders %s. P0 is the lord." % (self.rnd() + 1, self.rounds, p, self.res, self.bys)

    def over(self): return self.turn >= self.n * self.rounds
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
