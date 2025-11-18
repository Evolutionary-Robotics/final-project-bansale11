import glob, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from cartpole import CartPole
from model_factory import make_net

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["fnn","drnn"], required=True)
parser.add_argument("--indir", required=True)
parser.add_argument("--layers_fnn", type=str, default="4,16,16,2")
parser.add_argument("--layers_drnn", type=str, default="4,16,2")

parser.add_argument("--obs_noise", type=float, default=0.0, help="sensor noise σ")
parser.add_argument("--act_noise", type=float, default=0.0, help="action noise σ")
parser.add_argument("--init_pos", type=float, default=0.2, help="uniform init |x|≤init_pos")
parser.add_argument("--init_ang", type=float, default=0.2, help="uniform init |θ|≤init_ang")
parser.add_argument("--T", type=float, default=20.0, help="horizon (s)")
parser.add_argument("--dt", type=float, default=0.02, help="timestep (s)")
parser.add_argument("--seed", type=int, default=None, help="rng seed (optional)")
args = parser.parse_args()

# Choose architecture and parse layer sizes
if args.seed is not None:
    np.random.seed(args.seed)

layers = list(map(int, (args.layers_fnn if args.arch=="fnn" else args.layers_drnn).split(",")))

best_paths = sorted(glob.glob(os.path.join(args.indir, "best_*.npy")))
if not best_paths:
    raise SystemExit(f"No genomes found in {args.indir}/best_*.npy")
geno = np.load(best_paths[-1])

net = make_net(geno, arch=args.arch, layers=layers)
env = CartPole()
env.x = np.random.uniform(-args.init_pos, args.init_pos)
env.theta = np.random.uniform(-args.init_ang, args.init_ang)
if hasattr(net, "reset"): net.reset()

dt, T = args.dt, args.T
xs, thetas = [], []
steps = 0.0

for _ in range(int(T / dt)):
    s = env.state()
    xs.append(s[0]); thetas.append(s[2])
    s[2] *= 2.0; s[3] *= 2.0
    if args.obs_noise > 0.0:
        s = s + np.random.normal(0.0, args.obs_noise, size=s.shape)

    out = np.asarray(net.forward(s)).ravel()
    left  = (out[0] if out.size > 1 else out.item())
    right = (out[1] if out.size > 1 else 0.0)
    a = -1.0 if left > right else +1.0

    if args.act_noise > 0.0:
        a = np.clip(a + np.random.randn() * args.act_noise, -1.0, 1.0)

    r, done = env.step(dt, a)
    steps += r
    if done: break

print(f"Episode length: {steps:.1f} steps (obs σ={args.obs_noise}, act σ={args.act_noise})")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(xs); ax[0].set_title("Cart Position x"); ax[0].set_xlabel("Steps"); ax[0].set_ylabel("Position")
ax[1].plot(thetas); ax[1].set_title("Pole Angle θ (radians)"); ax[1].set_xlabel("Steps"); ax[1].set_ylabel("Angle")
for a in ax: a.grid(True, alpha=0.3)
plt.tight_layout()
suffix = f"_obs{args.obs_noise:.3f}_act{args.act_noise:.3f}"
outpath = os.path.join(args.indir, f"behavior_rollout{suffix}.png")
plt.savefig(outpath, dpi=150); plt.show()
print(f"[saved] {outpath}")
