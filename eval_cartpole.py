import glob, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from cartpole import CartPole
from model_factory import make_net

def rollout(net, T=20.0, dt=0.02, x0=0.0, th0=0.0, thd0=0.0,
            obs_noise=0.0, act_noise=0.0):
    env = CartPole()
    env.x = float(x0); env.theta = float(th0); env.theta_dot = float(thd0)
    if hasattr(net, "reset"): net.reset()

    steps = 0.0
    for _ in range(int(T/dt)):
        s = env.state()
        s[2] *= 2.0; s[3] *= 2.0
        if obs_noise > 0.0:
            s = s + np.random.normal(0.0, obs_noise, size=s.shape)

        out = np.asarray(net.forward(s)).ravel()
        a = -1.0 if (out[0] if out.size>1 else out.item()) > (out[1] if out.size>1 else 0.0) else +1.0
        if act_noise > 0.0:
            a = np.clip(a + np.random.randn()*act_noise, -1.0, 1.0)

        r, done = env.step(dt, a)
        steps += r
        if done or r == 0.0:
            break
    return steps

def evaluate_distribution(net, episodes=20, T=20.0, dt=0.02,
                          init_pos=0.2, init_ang=0.2,
                          obs_noise=0.20, act_noise=0.0, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        urand = lambda lo, hi: rng.uniform(lo, hi)
    else:
        urand = np.random.uniform

    scores = []
    for _ in range(episodes):
        x0  = urand(-init_pos, init_pos)
        th0 = urand(-init_ang,  init_ang)
        scores.append(rollout(
            net, T=T, dt=dt, x0=x0, th0=th0, thd0=0.0,
            obs_noise=obs_noise, act_noise=act_noise
        ))
    return np.array(scores, float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["fnn","drnn"], required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--layers_fnn", type=str, default="4,16,16,2")
    ap.add_argument("--layers_drnn", type=str, default="4,16,2")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--obs_noise", type=float, default=0.20)
    ap.add_argument("--act_noise", type=float, default=0.0)
    ap.add_argument("--hist_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--heat_T", type=float, default=15.0)
    ap.add_argument("--heat_dt", type=float, default=0.02)
    ap.add_argument("--heat_xmin", type=float, default=-0.5)
    ap.add_argument("--heat_xmax", type=float, default=0.5)
    ap.add_argument("--heat_thmin", type=float, default=-0.2)
    ap.add_argument("--heat_thmax", type=float, default=0.2)
    ap.add_argument("--heat_nx", type=int, default=25)
    ap.add_argument("--heat_nth", type=int, default=25)
    ap.add_argument("--heat_obs_noise", type=float, default=0.10)

    ap.add_argument("--train_like_eps", type=int, default=100)
    ap.add_argument("--train_like_T", type=float, default=10.0)
    ap.add_argument("--train_like_init_pos", type=float, default=0.2)
    ap.add_argument("--train_like_init_ang", type=float, default=0.2)
    ap.add_argument("--train_like_obs_noise", type=float, default=0.05)

    ap.add_argument("--hard_test_eps", type=int, default=100)
    ap.add_argument("--hard_test_T", type=float, default=15.0)
    ap.add_argument("--hard_test_init_pos", type=float, default=0.5)
    ap.add_argument("--hard_test_init_ang", type=float, default=0.2)
    ap.add_argument("--hard_test_obs_noise", type=float, default=0.10)

    args = ap.parse_args()

    best_paths = sorted(glob.glob(os.path.join(args.indir, "best_*.npy")))
    if not best_paths:
        raise SystemExit(f"No genomes found in {args.indir}/best_*.npy. Run training first.")
    geno = np.load(best_paths[-1])

    layers = list(map(int, (args.layers_fnn if args.arch=="fnn" else args.layers_drnn).split(",")))
    net = make_net(geno, arch=args.arch, layers=layers)

    scores = evaluate_distribution(
        net,
        episodes=args.episodes, T=20.0, dt=0.02,
        init_pos=0.2, init_ang=0.2,
        obs_noise=args.obs_noise, act_noise=args.act_noise,
        seed=args.seed
    )
    avg, std = float(np.mean(scores)), float(np.std(scores))
    print(f"\nCartPole test: {avg:.1f} ± {std:.1f} steps over {len(scores)} episodes "
          f"(obs σ={args.obs_noise}, act σ={args.act_noise})")
    print(f" {'Solved!' if avg >= 475 else 'Not yet (Gym threshold = 475)'}")

    plt.figure()
    plt.hist(scores, bins=args.hist_bins)
    plt.xlabel("Episode length (steps)"); plt.ylabel("Count")
    plt.title("CartPole performance distribution")
    plt.tight_layout()
    path = os.path.join(args.indir, "performance_histogram.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"[plot] {path}")

    x_grid  = np.linspace(args.heat_xmin, args.heat_xmax, args.heat_nx)
    th_grid = np.linspace(args.heat_thmin, args.heat_thmax, args.heat_nth)
    H = np.zeros((len(th_grid), len(x_grid)))
    for i, th0 in enumerate(th_grid):
        for j, x0 in enumerate(x_grid):
            H[i, j] = rollout(
                net, T=args.heat_T, dt=args.heat_dt, x0=x0, th0=th0, thd0=0.0,
                obs_noise=args.heat_obs_noise, act_noise=0.0
            )

    plt.figure(figsize=(6,5))
    im = plt.imshow(H, origin="lower",
                    extent=[x_grid[0], x_grid[-1], th_grid[0], th_grid[-1]],
                    aspect="auto", cmap="viridis_r", vmin=0, vmax=1000)
    plt.xlabel("x₀ (m)"); plt.ylabel("θ₀ (rad)")
    plt.title("Generalization heatmap (yellow = failure, dark = success)")
    plt.colorbar(im, label="Episode length (steps)")
    plt.tight_layout()
    path = os.path.join(args.indir, "generalization_heatmap_inverted.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"[plot] {path}")

    train_scores = evaluate_distribution(
        net, episodes=args.train_like_eps, T=args.train_like_T, dt=0.02,
        init_pos=args.train_like_init_pos, init_ang=args.train_like_init_ang,
        obs_noise=args.train_like_obs_noise, act_noise=0.0
    )
    test_scores  = evaluate_distribution(
        net, episodes=args.hard_test_eps, T=args.hard_test_T, dt=0.02,
        init_pos=args.hard_test_init_pos, init_ang=args.hard_test_init_ang,
        obs_noise=args.hard_test_obs_noise, act_noise=0.0
    )

    plt.figure(figsize=(6,4))
    plt.boxplot([train_scores, test_scores], labels=["Train-like", "Harder test"])
    plt.ylabel("Episode length (steps)")
    plt.title("Generalization: train vs. test distributions")
    plt.tight_layout()
    path = os.path.join(args.indir, "generalization_boxplot.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"[plot] {path}")

    print("\nGeneralization summary")
    print(f"  Train-like: mean={np.mean(train_scores):.1f} ± {np.std(train_scores):.1f}")
    print(f"  Harder test: mean={np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
    print(f"  Test success rate @1000 steps: {(np.mean(test_scores >= 1000.0)*100):.1f}%")

if __name__ == "__main__":
    main()
