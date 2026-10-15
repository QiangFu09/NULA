"""Train a mean-field network with NULA, NLA, or EM-NULA."""

import argparse
import json
import math
from pathlib import Path
import platform


def parse_args(default_methods=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=["nula", "nla", "em"], nargs="+", default=default_methods or ["nula", "nla", "em"])
    parser.add_argument("--particles", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.dimension, *args.particles) < 1:
        parser.error("epochs, samples, dimension and particle counts must be positive")
    for name in ["methods", "particles", "seeds"]:
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"{name} must be distinct")
    if min(args.seeds) < 0:
        parser.error("seeds must be nonnegative")
    return args


def result_path(directory, method, particles, seed):
    return directory / f"{method}_n{particles}_seed{seed}.json"


def main(default_methods=None):
    args = parse_args(default_methods)
    paths = [result_path(args.output_dir, method, n, seed) for method in args.methods for n in args.particles for seed in args.seeds]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Result exists: {existing[0]}. Choose another --output-dir or use --overwrite.")
    import torch
    from nula import Net, NULAOptimizer, EMNULAOptimizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(0)
    center = torch.rand(args.dimension)
    data = torch.rand(args.samples, args.dimension)
    target = torch.exp(-torch.sum((data - center) ** 2, dim=1) / (2 * args.dimension))
    data, target = data.to(args.device), target.to(args.device)
    cfg = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Separate the two sweeps to retain the original scripts' random-number order.
    groups = [[m for m in ["nula", "nla"] if m in args.methods], [m for m in ["em"] if m in args.methods]]
    for methods in groups:
        for seed in args.seeds:
            torch.manual_seed(seed)
            # The original scripts drew a candidate dataset here but trained on
            # the fixed loader created above. Retain that RNG consumption.
            torch.rand(args.samples, args.dimension)
            for particles in args.particles:
                for method in methods:
                    net = Net(N=particles, cut_off=10, activation_type="Tanh", input_dim=args.dimension).to(args.device)
                    cls = EMNULAOptimizer if method == "em" else NULAOptimizer
                    optimizer = cls(net.parameters(), particles, lr=0.01, lr_2=0.01, momentum=method != "nla", gamma=1, weight_decay=1e-4, noise_scale=math.sqrt(2e-4) if method == "em" else 0.001)
                    losses, regularized = [], []
                    for epoch in range(1, args.epochs + 1):
                        optimizer.zero_grad()
                        loss = ((net(data) - target) ** 2 / 2).mean()
                        loss.backward()
                        optimizer.step()
                        value = loss.item()
                        norm = sum(torch.sum(param.detach() ** 2).item() / particles for param in net.parameters())
                        if not math.isfinite(value + norm):
                            raise RuntimeError(f"Non-finite result: {method}, N={particles}, seed={seed}, epoch={epoch}")
                        losses.append(value)
                        regularized.append(value + norm * 1e-4)
                        if epoch == 1 or epoch % 1000 == 0:
                            print(f"{method}: N={particles}, seed={seed}, epoch={epoch}, loss={value:.6g}")
                    payload = {"method": method, "particles": particles, "seed": seed, "configuration": cfg, "versions": {"python": platform.python_version(), "torch": torch.__version__}, "train_loss": losses, "regularized_train_loss": regularized}
                    path = result_path(args.output_dir, method, particles, seed)
                    with path.open("w" if args.overwrite else "x") as handle:
                        json.dump(payload, handle, indent=2, allow_nan=False)
                        handle.write("\n")
                    print(f"Saved {path}")


if __name__ == "__main__":
    main()
