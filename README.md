# NULA: mean-field underdamped Langevin experiments

Research code for **Mean-field Underdamped Langevin Dynamics and its Spacetime
Discretization**, Qiang Fu and Ashia Camage Wilson, ICML 2024.

[Paper](https://proceedings.mlr.press/v235/fu24g.html)

This repository contains the mean-field neural network experiments comparing
the N-particle underdamped Langevin algorithm (NULA), its Euler–Maruyama
discretization (EM-NULA), and the overdamped N-particle Langevin algorithm (NLA).
The experiment fits a synthetic Gaussian target with a two-layer tanh network.
The number of hidden units is the number of particles.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements-tested.txt` pins the direct dependency versions used for the
validation checks.

CPU is the default. Use `--device cuda` with a CUDA-enabled PyTorch installation
to run on a GPU. No external data are required.

## Training

Check the installation with a small CPU run:

```bash
python train.py --particles 4 --seeds 0 --epochs 3 \
  --dimension 8 --samples 4 --output-dir outputs/check
```

Run the released experiment settings:

```bash
python train.py --methods nula nla em \
  --particles 256 512 1024 2048 --seeds 0 1 2 3 4 --epochs 10000
```

This trains 60 networks: three methods, four particle counts, and five seeds.
The defaults use 100 training points in dimension 1,000. The training data are
fixed; seeds govern model initialization and optimizer randomness. The output
contains one JSON file per method, particle count, and seed, with losses,
configuration, and software versions. Existing results require `--overwrite`.

The historical update coefficients are retained in `nula.py`. In particular,
NULA's coefficients and noise magnitudes are fixed by that implementation;
the repository does not expose a general stepsize sweep. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) before changing numerical updates.

## Figures

For Figure 1, compare all methods at a fixed particle count:

```bash
python plot.py --figure 1 --particles 256 --output figures/figure1_n256.pdf
```

Repeat with 512, 1,024, and 2,048 particles for the other panels. For Figure 2:

```bash
python plot.py --figure 2 --output figures/figure2.pdf
```

Lines show mean training loss across the selected seeds; shading shows the
minimum and maximum. The plotting command checks that all requested results
exist. Its `--input-dir` must match the training command's `--output-dir`.

## Code organization

| File | Purpose |
| --- | --- |
| `nula.py` | Network and the released numerical update rules |
| `train.py` | Synthetic data, training sweeps, and JSON results |
| `plot.py` | Method and particle-count comparisons |
| `tests/` | Optimizer, reproducibility, and command-line checks |

`UNLA and NLA.py`, `EM-UNLA.py`, `Figure1.py`, and `Figure2.py` remain as short
compatibility entry points. `UNLA` is the historical filename spelling of NULA.
Run commands from the repository root and use `--help` for available options.

## Reproducibility

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the data and RNG ordering,
loss definitions, and implementation details retained from the initial release.
The original pickle results are not included. This release writes JSON and
regenerates the figures from those files.

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{fu2024meanfield,
  title     = {Mean-field Underdamped Langevin Dynamics and its Spacetime Discretization},
  author    = {Fu, Qiang and Wilson, Ashia Camage},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  volume    = {235},
  pages     = {14175--14206},
  publisher = {PMLR},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/fu24g.html}
}
```

## License and attribution

Original contributions are available under the [MIT License](LICENSE).
The network and optimizers are adapted from
[SeiruGanki/Uniform-in-time-propagation-of-chaos-for-kinetic-mean-field-Langevin-dynamics](https://github.com/SeiruGanki/Uniform-in-time-propagation-of-chaos-for-kinetic-mean-field-Langevin-dynamics).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the scope of that grant.
