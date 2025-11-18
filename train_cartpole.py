import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from cartpole import CartPole
from model_factory import make_net, genesize_for
from ea2 import Microbial

# Fitness
class CartPoleFitness:
    def __init__(self, layers, arch="drnn", train_noise_std=0.0):
        self.layers = layers
        self.arch = arch
        self.dt = 0.02
        self.T = 5.0
        self.episodes = 2
        self.base_noise = train_noise_std  # sensor noise added to observed state
        self.init_noise = 0.05

    def schedule(self, g, G):
        frac = g / max(1, G - 1)
        self.T = 5.0 + frac * 15.0
        self.episodes = 2 if frac < 0.5 else 4
        self.init_noise = 0.05 + frac * 0.15

    def __call__(self, geno):
        net = make_net(geno, self.arch, self.layers)
        total = 0.0

        for _ in range(self.episodes):
            env = CartPole()
            env.reset(noise=self.init_noise)
            if hasattr(net, "reset"): net.reset()  # useful for DRNN

            steps = 0.0
            for _ in range(int(self.T / self.dt)):
                s = env.state()
                s[2] *= 2.0; s[3] *= 2.0

                # sensor noise on observation
                if self.base_noise > 0.0:
                    s = s + np.random.randn(*s.shape) * self.base_noise

                out = np.asarray(net.forward(s)).ravel()
                a = -1.0 if (out[0] if out.size > 1 else out.item()) > (out[1] if out.size > 1 else 0.0) else +1.0
                r, done = env.step(self.dt, a)
                steps += r
                if done: break

            # keep shaping
            s = env.state()
            steps += 0.05 * (2.4 - abs(s[0])) + 0.1 * (0.209 - abs(s[2]))
            total += steps

        return total / self.episodes

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["fnn","drnn"], default="drnn")
    ap.add_argument("--train_noise", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--gens", type=int, default=140)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--mutstd", type=float, default=0.02)
    ap.add_argument("--recomb", type=float, default=0.5)
    ap.add_argument("--deme", type=int, default=18)
    ap.add_argument("--tpg_frac", type=float, default=0.8)
    ap.add_argument("--layers_fnn", type=str, default="4,16,16,2")
    ap.add_argument("--layers_drnn", type=str, default="4,16,2")
    args = ap.parse_args()

    layers = list(map(int, (args.layers_fnn if args.arch=="fnn" else args.layers_drnn).split(",")))
    genesize = genesize_for(args.arch, layers)
    tpg = int(args.pop * args.tpg_frac)

    outdir = f"cp_runs/{args.arch}_noise{args.train_noise:.3f}"
    os.makedirs(outdir, exist_ok=True)

    fitness = CartPoleFitness(layers, arch=args.arch, train_noise_std=args.train_noise)
    def fitness_fn(g): return fitness(g)

    all_best = []
    for k in range(args.runs):
        print(f"\n=== {args.arch.upper()} | train_noise={args.train_noise:.3f} | Run {k+1}/{args.runs} ===")
        ea = Microbial(
            fitness_fn=fitness_fn,
            popsize=args.pop,
            genesize=genesize,
            recomb=args.recomb,
            mutstd=args.mutstd,
            deme=args.deme,
            generations=args.gens,
            seed=1000 + k,
            tournaments_per_gen=tpg
        )
        def sched(g): fitness.schedule(g, args.gens)
        best, bh, ah = ea.run(on_gen=sched)

        np.save(os.path.join(outdir, f"best_{k}.npy"), best)
        np.savez(os.path.join(outdir, f"hist_{k}.npz"), best=bh, avg=ah)
        all_best.append(bh)

    # overlay plot
    plt.figure()
    for b in all_best: plt.plot(b, alpha=0.8)
    plt.xlabel("Generation"); plt.ylabel("Best fitness (steps)")
    plt.title(f"{args.arch.upper()} best fitness | train noise={args.train_noise:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fitness_overlay.png"), dpi=150)
    print("Saved runs to:", outdir)

if __name__ == "__main__":
    main()
