# Fast microbial EA with demes
import numpy as np

class Microbial:
    def __init__(self, fitness_fn, popsize, genesize, recomb=0.5, mutstd=0.02,
                 deme=16, generations=120, seed=None, tournaments_per_gen=None):
        if seed is not None: np.random.seed(seed)
        self.fitness_fn = fitness_fn
        self.popsize = popsize
        self.genesize = genesize
        self.recomb = recomb
        self.mutstd = mutstd  # std dev of Gaussian mutation
        self.deme = max(1, int(deme/2))
        self.generations = generations
        self.tournaments_per_gen = tournaments_per_gen or popsize
        # Initialize population uniformly in [-1, 1]
        self.pop = np.random.rand(popsize, genesize) * 2 - 1 
        # Initial fitness values
        self.fit = np.array([fitness_fn(ind) for ind in self.pop])
        self.best_hist = np.zeros(generations)
        self.avg_hist = np.zeros(generations)
        self.best_ind = self.pop[np.argmax(self.fit)]

    def _stats(self, g):
        self.best_ind = self.pop[np.argmax(self.fit)]
        self.best_hist[g] = float(np.max(self.fit))
        self.avg_hist[g] = float(np.mean(self.fit))

    def run(self, on_gen=None):
        for g in range(self.generations):
            if on_gen is not None: on_gen(g)
            for _ in range(self.tournaments_per_gen):
                a = np.random.randint(0, self.popsize)
                b = np.random.randint(a-self.deme, a+self.deme) % self.popsize
                while b == a: b = np.random.randint(a-self.deme, a+self.deme) % self.popsize
                 # Identify winner w and loser l by fitness
                w, l = (a, b) if self.fit[a] > self.fit[b] else (b, a)
                mask = (np.random.rand(self.genesize) >= self.recomb).astype(float)
                child = mask*self.pop[l] + (1-mask)*self.pop[w]
                child += np.random.normal(0.0, self.mutstd, size=self.genesize)
                child = np.clip(child, -1, 1)
                self.pop[l] = child
                self.fit[l] = self.fitness_fn(child)
            self._stats(g)
            if g % 5 == 0 or g == self.generations-1:
                print(f"  Gen {g}/{self.generations-1} | Best {self.best_hist[g]:.1f}  Avg {self.avg_hist[g]:.1f}", flush=True)
        return self.best_ind, self.best_hist, self.avg_hist
