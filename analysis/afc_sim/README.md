# afc_sim

Offline simulation framework for comparing Adaptive Feedback Cancellation
(AFC) algorithms against real measured hearing-aid feedback-path impulse
responses. Built to answer one question at a time as algorithms are added:
given a real feedback path, how fast does each algorithm converge, how good
is its steady-state estimate, and how much stable gain does it actually buy
over doing nothing?

This document is a map of the package, not a tutorial on AFC theory. Several
design decisions have "why" explanations inline as code comments/docstrings;
this README points you to where those live rather than repeating them.

## Where the data comes from

`conditions.py` loads impulse responses from an `.npz` produced by
`mls_impulse.py --save_ir` (the MLS deconvolution script elsewhere in
`analysis/`). One recording session yields many individual MLS periods, each
one a noisy measurement of the same physical feedback path. `conditions.py`
splits these into disjoint **train** and **eval** ("held-out") pools:

- Training periods are used to fit a PCA model of "what feedback paths from
  this recording setup tend to look like" (`pca_training.py`).
- Held-out periods become the **ground truth** for each simulated "condition"
  — a different held-out period is swapped in every time the simulation
  switches to a new epoch, standing in for the feedback path changing (a
  hand moving near the ear, a door opening, etc.).

Caveat carried everywhere in this code: with only one physical recording
session, held-out periods differ from each other only by measurement noise,
not by genuinely different acoustic conditions. This validates that the
framework works end-to-end, not that any algorithm's real-world advantage is
proven — that needs recordings from actually distinct physical setups.

`--n-average N` (default 1) averages every `N` consecutive, non-overlapping
raw periods into one condition before anything else happens, trading fewer
total candidate conditions for lower measurement noise per condition.

## Module layout

| File | Role |
|---|---|
| `conditions.py` | Loads/splits impulse responses; `zero_lag_window()` — the single most important convention in this codebase (see below) |
| `signals.py` | Generates the forward/excitation signal that drives each epoch: white, pink, colored (AR(1)), real WAV, or `ar_fit` (stationary noise shaped like a real recording's envelope) |
| `feedback_path.py` | Convolves the excitation against the active ground-truth IR to produce the contaminated mic signal (open-loop only) |
| `pca_training.py` | Thin wrapper around `train_pca_feedback_model.py`'s `fit_pca`/`write_pca_csv`, using `conditions.py`'s windowing convention |
| `metrics.py` | misalignment (NPM), ERLE, convergence time, and MSG/ASG (Maximum/Added Stable Gain, via the Nyquist criterion applied to the IR's frequency response) |
| `simulate.py` | The main loop: drives every algorithm through the same epoch schedule, snapshotting metrics at a fixed cadence |
| `closed_loop.py` | Genuine per-sample closed-loop simulation (algorithm's own output is acoustically coupled back) — secondary/validation tool, see below |
| `results_io.py` | Writes `results.csv` + `manifest.json`, and one plot file per metric |
| `run_experiment.py` | The CLI entry point — see Usage below |
| `gain_amplitude_sweep.py` | Standalone script mapping the usable (signal amplitude, gain) region for closed-loop testing |
| `algorithms/` | Each AFC variant — see table below |
| `tests/test_afc_sim.py` | Regression suite, no pytest dependency — run with `python -m afc_sim.tests.test_afc_sim` |

### The one convention to understand: `zero_lag_window`

Every algorithm's regressor has a fixed 1-sample causal lag (`efbp[m]`
multiplies `spk[i-1-m]`, never `spk[i]` — a sample can't depend on itself).
`FeedbackPath`'s convolution convention is `ir[k]` = gain from `spk[i-k]`,
*including* `ir[0]`, the same-instant tap no causal algorithm can ever
reach. So the open-loop ground-truth target is `ir[1:afl+1]`, not
`ir[0:afl]` — a 1-sample shift. `conditions.zero_lag_window(ir, afl, lag=1)`
is this shift; `lag=0` is used instead for closed-loop ground truth, because
`closed_loop.py`'s own per-sample loop already bakes the causal lag into its
convolution, so applying the open-loop shift *again* on top double-shifts
the target. Getting this wrong previously produced a real, confirmed bug
(reported +7dB misalignment — worse than no cancellation at all — for a
correctly-converged closed-loop filter). If you add a new ground-truth
comparison anywhere, get the `lag` right for whichever loop mode you're in.

## Algorithm variants

All registered via `afc_sim.algorithms.registered_names()`, selected with
`--variants name1,name2,...`:

| Name | What it is |
|---|---|
| `pemafc_f` | Full-rank (afl taps) NLMS with PEM-AFC's fixed first-order pre-emphasis whitening (mode F, `[1, -alpha]`) |
| `pemafc_a` | Same, but the whitening AR coefficients are themselves adaptively re-estimated from the error signal every `ar_block_len` samples (mode A) |
| `nlms` | Plain NLMS, **no whitening at all** — `pemafc_f` with `alpha` forced to 0, which makes the whitening filter an exact identity (verified bit-exact against a hand-coded reference) |
| `pemafc_pca_f` / `pemafc_pca_a` | PCA-constrained: `efbp = mean_ir + c @ components`, only the `K`-dim `c` adapts (mode F/A front end, same as above) |
| `nlms_pca` | PCA-constrained + no whitening, same relationship to `pemafc_pca_f` as `nlms` has to `pemafc_f` |
| `soft_pca` | Full-rank NLMS, but after every sample the result is partially shrunk back toward the PCA subspace: `efbp = efbp_raw - soft_pca_alpha * (off-subspace component)`. One coefficient vector, continuously biased — see `algorithms/pemafc_soft_pca.py` |
| `blend_pca` | **Two fully independent filters** (`pemafc_f`-equivalent and `pemafc_pca_f`-equivalent, each adapting on its own error as if it were the sole canceller), combined only at the output: `efbp = (1-w)*efbp_full + w*efbp_pca`. Fixed `w` |
| `blend_pca_dynamic` | Same two independent filters, but `w` is itself adapted by gradient descent on the blended output's own error (Arenas-García et al.'s "combination of adaptive filters," see References) |
| `null_afc` | No cancellation at all (`e = mic` always) — baseline for "how much does any algorithm help," and the correct topology for empirically validating the analytical `MSG_without_AFC` formula |

`pemafc_f`/`pemafc_a` and their PCA/NLMS variants all default to **mode F**
unless the variant name says `_a`. There is a standing project preference
(not a code default) to use mode F, not mode A, when testing PCA-constrained
variants specifically — see memory notes if picking defaults for a new run.

### Choosing between `soft_pca` and `blend_pca[_dynamic]`

Both exist to get "PCA's fast convergence without being stuck at PCA's worse
floor forever," but they're mechanistically different:

- `soft_pca` is cheaper (one coefficient vector) and has one tunable knob
  (`soft_pca_alpha`, or equivalently a decay time constant `tau`).
- `blend_pca_dynamic` genuinely runs two independent adaptive processes and
  is more expensive (~2-3x one algorithm's per-sample arithmetic), but each
  side gets to use its own best-suited update rule rather than one state
  being geometrically projected. On the one real dataset tested so far,
  neither uniformly beats the other — they land at different points on the
  speed/accuracy tradeoff depending on the condition.

Real-device feasibility for `blend_pca*`: the two filters share one copy of
the raw sample history and (since both default to mode F) one whitening
filter. The only genuinely duplicated state is the PCA side's small
`K`-dimensional coefficient vector — the big `mean_ir`/`components` table is
a single fixed constant, not duplicated per filter. See
`algorithms/blend_pca.py`'s module docstring for the full cost breakdown.

## Open-loop vs. closed-loop (`--loop-mode`)

**Open-loop (default)**: an independent excitation signal is convolved once
against the known ground-truth IR to produce the whole epoch's mic signal
in advance. Fast, and the standard way to measure misalignment/ERLE/MSG/ASG
convergence. No real acoustic loop, no gain, can't diverge.

**Closed-loop**: the algorithm's own output (after `--gain-db` and clipping
at `--saturation-limit`) is genuinely, causally fed back sample-by-sample —
this is the only mode that can actually go unstable/howl. Much slower
(`closed_loop.py` is a literal per-sample Python loop). Kept as a secondary
tool: validating the analytical MSG formula against real observed
instability, and studying nonlinear/saturation effects open-loop can't see.
`gain_amplitude_sweep.py` is the dedicated tool for mapping the usable
(amplitude, gain) region in this mode.

## Metrics (`metrics.py`)

- **Misalignment (NPM)**: `20*log10(||true_ir - efbp|| / ||true_ir||)`.
  0dB = as bad as guessing zero. Negative = genuine improvement. Can
  legitimately be *positive* (worse than guessing zero) for a poorly-excited
  or diverging filter.
- **ERLE**: `10*log10(E[mic^2] / E[e^2])`. How many dB of the contaminating
  signal got removed, in actual residual power — not the same as
  misalignment, and the two can diverge (e.g. a drifting-but-still-partially-
  cancelling filter).
- **MSG (Maximum Stable Gain)**: computed directly from an impulse response's
  frequency response via the Nyquist criterion — the largest magnitude among
  all frequencies where phase is exactly 0 deg (mod 360). No simulation
  needed. `MSG_without_AFC` = raw feedback path; `MSG_with_AFC(t)` = the
  *residual* (`true_ir - efbp(t)`) — what's still actually there once the
  algorithm's estimate is subtracted out.
- **ASG**: `MSG_with_AFC(t) - MSG_without_AFC`. Can go negative even when
  misalignment is improving — broadband (L2) error reduction doesn't
  guarantee the single worst-case frequency gets better too, especially for
  subspace-constrained algorithms with less freedom to fix every frequency
  at once. Verified correct via independent root-finding cross-check; this
  is a real property of the metric, not a bug.

## Signal types (`--signal`)

| Kind | What |
|---|---|
| `white` | Gaussian white noise. Structurally *disadvantages* mode F's fixed pre-emphasis (nothing for it to correct) |
| `pink` | Cheap integrate-then-highpass approximation, not a clean spectral match to anything in particular |
| `colored` | AR(1), pole = `--signal-color-pole` (default 0.9, matching PEMAFCFull's default `alpha`) — exactly the coloration mode F's fixed pre-emphasis is designed to invert |
| `wav` | Replay a real WAV file (must match the simulation's sample rate exactly — no resampling) |
| `ar_fit` | Fits an LPC envelope (`--ar-fit-order`) to a segment of a real WAV (`--ar-fit-start-s`/`--ar-fit-duration-s`), then generates **stationary** synthetic noise with that same long-term spectral shape. Same coloration a real recording has, without its pauses/dynamic-range swings that otherwise make convergence curves noisy and hard to read |

## Running an experiment

```
python -m afc_sim.run_experiment ../data/self/AUDIO001_ir_hp100.npz \
    --afl 512 --variants pemafc_f,pemafc_pca_f,nlms,nlms_pca \
    --epoch-duration 10 --n-eval 4 --mu 0.03 \
    --out-dir results/my_run
```

Flags are grouped roughly as: condition loading (`--n-train`, `--n-eval`,
`--n-average`, `--min-quality-percentile`, `--seed`), excitation signal
(`--signal` and its kind-specific sub-flags), algorithm selection
(`--variants` and each variant family's own flags —
`--soft-pca-taus`/`--soft-pca-alphas`, `--blend-weights`,
`--blend-dynamic-*`), shared algorithm hyperparameters (`--mu`, `--delta`,
`--alpha`, `--ar-order`, `--ar-block-len`, `--ar-reg`), PCA training
(`--n-components-train`, `--var-explained`, `--device-max-components`,
`--n-components-active`), loop mode (`--loop-mode`, `--gain-db`,
`--saturation-limit`), and metrics cadence (`--metrics-interval-s`,
`--msg-interval-s`, `--msg-fft-oversample`). Run `--help` for the full,
current list with explanations — every flag has one.

## Output files

Each `--out-dir` gets:

- `results.csv` — one row per metrics snapshot per (algorithm, epoch);
  long-format, every column repeated rather than normalized out, by design
  (see `save_manifest`'s docstring on why).
- `manifest.json` — full CLI args, timestamp, train/eval condition IDs,
  convergence times, saturated fractions, `MSG_without_AFC` per
  (algorithm, epoch, condition).
- `misalignment_db.png`, `misalignment_linear.png`, `erle.png`, `msg.png`,
  `asg.png` — one metric per file, one line per algorithm, dashed vertical
  lines at condition switches. `msg.png` additionally shows each condition's
  `MSG_without_AFC` as a horizontal dotted black reference line (not tied to
  any algorithm's color, since it doesn't depend on which algorithm you're
  looking at).
- `diagnostics_<key>.png` — only written if some algorithm reports extra
  state via `get_diagnostics()` (currently just `blend_pca_dynamic`'s
  `blend_weight`). Not written at all if nothing has anything to show.

Note: `diagnostics_*` data is **not** persisted into `results.csv` — it only
exists in the plot. Regenerating a diagnostics plot without the original
`TrialResult` objects in memory requires re-running the simulation.

## Tests

```
python -m afc_sim.tests.test_afc_sim
```

No pytest dependency (none is installed in this environment) — plain
functions, each asserts or prints `PASS`. Several of these are regression
tests for real, previously-confirmed bugs (chunk-size-dependent results,
the closed-loop off-by-one ground truth, a corrupted `--help` string from an
unescaped `%`) — if you touch the files they cover, run this before trusting
a result.

## References

- MSG/ASG: standard Nyquist/Barkhausen-criterion formulation from the
  hearing-aid AFC literature.
- `blend_pca_dynamic`'s gradient-adapted mixing weight: J. Arenas-García,
  A. R. Figueiras-Vidal, A. H. Sayed, "Mean-square performance of a convex
  combination of two adaptive filters," *IEEE Transactions on Signal
  Processing*, 2006; and the survey J. Arenas-García, L. A. Azpicueta-Ruiz,
  M. T. M. Silva, V. H. Nascimento, A. H. Sayed, "Combinations of Adaptive
  Filters," *IEEE Signal Processing Magazine*, Jan. 2016, vol. 33,
  pp. 120-140 (freely available at
  [arXiv:2112.12245](https://arxiv.org/abs/2112.12245)).
