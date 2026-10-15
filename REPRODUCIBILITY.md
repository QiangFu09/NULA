# Reproducibility notes

## Experiment

The network has one tanh hidden layer, N hidden units, and a scalar output
divided by N. Output-layer weights are clamped to [-10, 10] before the forward
pass. The target is exp(-||x-a||²/(2d)), with a and the training inputs sampled
uniformly in [0, 1]^d. Defaults are d=1,000, 100 samples, and 10,000 full-batch
updates. N ranges over 256, 512, 1,024, and 2,048.

The original loader was constructed once under Torch seed 0. Later per-seed
datasets were created but were not attached to that loader. Training therefore
used the same data across seeds. `train.py` retains those fixed data and the
otherwise unused per-seed random draw so the CPU random-number sequence follows
the original full sweep. Run all methods, particles, and seeds in the documented
order when comparing with the released settings. Selecting a subset of methods
or particle counts changes later initializations because the RNG is consumed
sequentially. EM-NULA uses a separate sweep, matching the two original scripts.

CPU is now the explicit default. CUDA remains optional; CPU and CUDA random
number streams and floating-point results need not match.

## Retained numerical behavior

`nula.py` preserves the released updates rather than substituting a new
discretization. Deprecated `Tensor.add_(scalar, tensor)` calls are written using
the equivalent `alpha=` signature. The EM optimizer reads gamma from its
parameter group instead of a module global; its default experiment value is 1.

Several historical details matter when interpreting or extending this code:

- NULA uses fixed position/gradient coefficients 0.0001 and 0.02, a factor 0.99,
  and independent 0.001-scale Gaussian noise draws.
- The NULA branch computes a new local velocity tensor after updating the
  parameters without storing that new tensor back in `momentum_buffer`.
  This behavior is retained. Changing it changes the experiment.
- In NLA, the original expression `lr ** 1 / 2` evaluates to lr/2. The code now
  spells it as `lr / 2`; it has not been changed to sqrt(lr).
- The EM momentum branch uses a fixed 0.001 noise scale even though its optimizer
  accepts a noise-scale argument. This remains the released behavior.

These observations describe the released implementation. Its numerical behavior
should be distinguished from implementing the paper's algorithms anew from
their mathematical definitions.

## Losses and plots

`train_loss` is half the mean squared error, evaluated before the update at each
epoch. `regularized_train_loss` adds the post-update squared parameter norm,
scaled by 1e-4/N, following the original diagnostic. It is not a separate
evaluation of the complete objective at a single iterate.

The figures use arithmetic means and minimum/maximum envelopes across the
selected seeds. These envelopes are ranges, not confidence intervals. Epoch
indices follow the original zero-based plotting convention; index zero is
outside the visible logarithmic axis.

## Validation and output

Imports no longer train models. The original filenames forward to the new
commands. Training writes JSON instead of pickle, including individual curves,
configuration, and software versions. The original result files are not bundled.

Short CPU checks use Python 3.12.3, PyTorch 2.5.1, NumPy 2.1.2, and Matplotlib
3.10.1. See `requirements-tested.txt` for direct version pins. The original
environment was not locked; full paper experiments have not been rerun during
this reorganization.
