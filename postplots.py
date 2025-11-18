import os, glob, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from model_factory import make_net
from cartpole import CartPole

def load_histories(indir):
    paths = sorted(glob.glob(os.path.join(indir, "hist_*.npz")))
    runs = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        runs.append({"best": z["best"], "avg": z["avg"]})
    return runs

def list_best_genomes(indir):
    # return list of (gen_idx, path) sorted by gen
    best_paths = sorted(glob.glob(os.path.join(indir, "best_*.npy")))
    out = []
    for p in best_paths:
        bn = os.path.basename(p)
        try:
            gen = int(bn.split("_")[1].split(".")[0])
        except Exception:
            gen = -1
        out.append((gen, p))
    out.sort(key=lambda t: t[0])
    return out

def latest_best_genome(indir):
    lst = list_best_genomes(indir)
    if not lst:
        return None
    return np.load(lst[-1][1])

def curriculum_max_steps(G, dt=0.02, T0=5.0, T1=20.0):
    gens = np.arange(G)
    frac = gens / max(1, G - 1)
    T = T0 + frac * (T1 - T0)
    return T / dt

def plot_learning_overlay(indir, title=None):
    runs = load_histories(indir)
    if not runs:
        print(f"[warn] no hist_*.npz in {indir}")
        return

    bests = np.stack([r["best"] for r in runs], axis=0)
    G = bests.shape[1]
    gens = np.arange(G)
    max_steps = curriculum_max_steps(G)
    norm_bests = bests / max_steps[None, :]
    mean_norm = norm_bests.mean(axis=0)
    std_norm  = norm_bests.std(axis=0)
    ci95_norm = 1.96 * std_norm / max(1, np.sqrt(norm_bests.shape[0]))

    plt.figure(figsize=(8,5))
    for nb in norm_bests:
        plt.plot(gens, nb, alpha=0.20, linewidth=1)
    plt.plot(gens, mean_norm, linewidth=2.0, label="Mean best (normalized)")
    plt.fill_between(gens,
                     mean_norm - ci95_norm,
                     mean_norm + ci95_norm,
                     alpha=0.25, label="95% CI")
    plt.xlabel("Generation")
    plt.ylabel("Fraction of curriculum horizon survived")
    plt.ylim(0, 1.05)
    plt.title(title or f"Learning curves — {os.path.basename(indir)} (normalized)")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    path = os.path.join(indir, "fitness_overlay_normalized.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.show()
    print(f"[plot & shown] {path}")

    max_mean = float(mean_norm.max())
    thresh   = 0.98 * max_mean
    k = int(np.argmax(mean_norm >= thresh)) if np.any(mean_norm >= thresh) else G
    k = max(10, min(k + 5, G))

    plt.figure(figsize=(8,5))
    for nb in norm_bests:
        plt.plot(gens[:k], nb[:k], alpha=0.25, linewidth=1)
    plt.plot(gens[:k], mean_norm[:k], linewidth=2.0, label="Mean best (normalized)")
    plt.fill_between(gens[:k],
                     (mean_norm - ci95_norm)[:k],
                     (mean_norm + ci95_norm)[:k],
                     alpha=0.25, label="95% CI")
    plt.xlabel("Generation")
    plt.ylabel("Fraction of curriculum horizon survived")
    plt.ylim(0, 1.05)
    plt.title((title or f"Learning curves — {os.path.basename(indir)}")
              + " (normalized, early zoom)")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    path_zoom = os.path.join(indir, "fitness_overlay_normalized_early_zoom.png")
    plt.tight_layout(); plt.savefig(path_zoom, dpi=150); plt.show()
    print(f"[plot & shown] {path_zoom}")
    aucs = np.trapz(norm_bests, axis=1)
    print(f"Normalized AUC mean±std across runs: {aucs.mean():.3f} ± {aucs.std():.3f} (R={norm_bests.shape[0]})")

def evaluate_distribution(net, episodes=50, T=20.0, dt=0.02,
                          init_pos=0.2, init_ang=0.2,
                          obs_noise=0.0, act_noise=0.0):
    scores = []
    if hasattr(net, "reset"): net.reset()
    for _ in range(episodes):
        env = CartPole()
        env.x = np.random.uniform(-init_pos, init_pos)
        env.theta = np.random.uniform(-init_ang, init_ang)

        steps = 0.0

        for _ in range(int(T/dt)):
            s = env.state()
            s[2] *= 2.0; s[3] *= 2.0
            if obs_noise > 0.0:
                s = s + np.random.normal(0.0, obs_noise, size=s.shape)

            out = np.asarray(net.forward(s)).ravel()
            a_score_left  = (out[0] if out.size > 1 else out.item())
            a_score_right = (out[1] if out.size > 1 else 0.0)
            a = -1.0 if a_score_left > a_score_right else +1.0

            if act_noise > 0.0:
                a = np.clip(a + np.random.randn()*act_noise, -1.0, 1.0)

            r, done = env.step(dt, a)
            steps += r
            if done: break
        scores.append(steps)
    return np.array(scores, float)

def robustness_curve(indir, arch, layers, sigmas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                     episodes=50, mode="sensor"):
    g = latest_best_genome(indir)
    if g is None:
        print(f"[warn] no best_*.npy in {indir}"); return
    net = make_net(g, arch=arch, layers=layers)

    means, stds = [], []
    for s in sigmas:
        if mode == "sensor":
            sc = evaluate_distribution(net, episodes=episodes, obs_noise=s, act_noise=0.0)
        elif mode == "action":
            sc = evaluate_distribution(net, episodes=episodes, obs_noise=0.0, act_noise=s)
        else:
            raise ValueError("mode must be 'sensor' or 'action'")
        means.append(float(np.mean(sc))); stds.append(float(np.std(sc)))
        print(f"{os.path.basename(indir)} | {mode} σ={s:.2f} → {means[-1]:.1f} ± {stds[-1]:.1f}")

    x = np.array(sigmas, float)
    plt.figure(); plt.plot(x, means, marker="o")
    plt.xlabel(f"Test {mode} noise σ"); plt.ylabel("Mean episode length (steps)")
    plt.title(f"Robustness ({mode}) — {os.path.basename(indir)}")
    path = os.path.join(indir, f"robustness_{mode}_curve_post.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.show()
    with open(os.path.join(indir, f"robustness_{mode}_stats.json"), "w") as f:
        json.dump({"sigmas": list(map(float, x)), "means": means, "stds": stds}, f, indent=2)
    print(f"[plot & shown] {path}")

def validate_best_over_gens(indir, arch, layers,
                            episodes=30, obs_noise=0.0, act_noise=0.0):
    best_list = list_best_genomes(indir)
    if not best_list:
        print(f"[warn] no best_*.npy in {indir}"); return

    gens, means, stds = [], [], []
    for gen, path in best_list:
        g = np.load(path)
        net = make_net(g, arch=arch, layers=layers)
        sc = evaluate_distribution(net, episodes=episodes,
                                   obs_noise=obs_noise, act_noise=act_noise)
        gens.append(gen)
        means.append(float(np.mean(sc)))
        stds.append(float(np.std(sc)))

    gens = np.array(gens)
    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(8,5))
    plt.plot(gens, means, linewidth=2.0, label="Noisy validation (mean)")
    plt.fill_between(gens, means-1.96*stds/np.sqrt(max(1,len(gens))),
                     means+1.96*stds/np.sqrt(max(1,len(gens))), alpha=0.25, label="95% CI")
    plt.xlabel("Generation"); plt.ylabel("Episode length (steps)")
    plt.title(f"Noisy validation — {os.path.basename(indir)} (obs σ={obs_noise}, act σ={act_noise})")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    out = os.path.join(indir, "noisy_validation_curve.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.show()
    print(f"[plot & shown] {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, nargs="+", help="one or more cp_runs/* condition folders")
    ap.add_argument("--arch", choices=["fnn","drnn"], required=True)
    ap.add_argument("--layers_fnn", type=str, default="4,16,16,2")
    ap.add_argument("--layers_drnn", type=str, default="4,16,2")

    ap.add_argument("--val_obs_noise", type=float, default=0.0, help="obs noise for noisy validation curve")
    ap.add_argument("--val_act_noise", type=float, default=0.0, help="action noise for noisy validation curve")
    ap.add_argument("--val_episodes", type=int, default=30)

    ap.add_argument("--robust_mode", choices=["sensor","action"], default="sensor")
    ap.add_argument("--robust_sigmas", type=str, default="0.0,0.05,0.1,0.15,0.2,0.3", help="comma list")
    ap.add_argument("--robust_episodes", type=int, default=50)

    args = ap.parse_args()

    for d in args.indir:
        layers = list(map(int, (args.layers_fnn if args.arch=="fnn" else args.layers_drnn).split(",")))
        plot_learning_overlay(d)

        # Noisy validation across generations
        if args.val_obs_noise > 0.0 or args.val_act_noise > 0:
            validate_best_over_gens(
                d, arch=args.arch, layers=layers,
                episodes=args.val_episodes,
                obs_noise=args.val_obs_noise, act_noise=args.val_act_noise
            )

        # Robustness sweep on latest best
        sigmas = tuple(map(float, args.robust_sigmas.split(",")))
        robustness_curve(
            d, arch=args.arch, layers=layers,
            sigmas=sigmas, episodes=args.robust_episodes,
            mode=args.robust_mode
        )

if __name__ == "__main__":
    main()
