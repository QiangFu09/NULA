"""Generate the method comparison (Figure 1) or particle comparison (Figure 2)."""

import argparse
import json
from pathlib import Path


def main(default_figure=1):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=[1, 2], default=default_figure)
    parser.add_argument("--particles", type=int, default=2048, help="Particle count for Figure 1")
    parser.add_argument("--particle-counts", type=int, nargs="+", default=[256, 512, 1024, 2048], help="Counts for Figure 2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from train import result_path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = [(m, args.particles, label) for m, label in [("nula", "NULA"), ("nla", "NLA"), ("em", "EM-NULA")]] if args.figure == 1 else [("nula", n, f"N = {n}") for n in reversed(args.particle_counts)]
    fig, ax = plt.subplots(figsize=(5, 4))
    reference = None
    for method, particles, label in comparisons:
        curves = []
        for seed in args.seeds:
            path = result_path(args.input_dir, method, particles, seed)
            if not path.is_file():
                parser.error(f"Missing {path}; run train.py for all requested methods, particles and seeds first")
            result = json.loads(path.read_text())
            cfg = result["configuration"]
            setup = tuple(cfg[key] for key in ["epochs", "samples", "dimension", "device"])
            if reference is not None and setup != reference:
                parser.error("Input files use different data, epoch or device configurations")
            reference = setup
            curves.append(result["train_loss"])
        data = np.asarray(curves)
        # The original figures use zero-based epoch indices on a logarithmic axis.
        x = np.arange(data.shape[1])
        line, = ax.loglog(x, data.mean(axis=0), label=label)
        ax.fill_between(x, data.min(axis=0), data.max(axis=0), color=line.get_color(), alpha=0.2, linewidth=0)
    ax.set(xlabel="Epoch index", ylabel="Training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output = args.output or Path("figures") / (f"figure1_n{args.particles}.pdf" if args.figure == 1 else "figure2.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
