# Offline AFC simulation framework (PEMAFC F/A vs. PCA-constrained)

## Context

The goal is an offline Python simulation that drives the PEMAFC adaptive
feedback-cancellation algorithm (as implemented in Tympan_Library, both its
"F" and "A" whitening modes) and the PCA-constrained variant against
measured feedback-path impulse responses, switching the active impulse
response partway through to see how each algorithm re-converges/tracks a
changing feedback path. The user explicitly wants this extensible ("I will
also be evaluating more than just the PCA model in the future") and is fine
with reorganizing/rewriting existing analysis code where it helps.

Two rounds of research (two parallel Explore agents, then a Plan-agent design
review) grounded this in the actual C++ source and the actual data on disk
rather than assumption. Key findings that shape the design:

1. **There is one C++ class, not two.** `AudioFeedbackCancelPEMAFC_F32` has
   an internal `PEMAFCMode::{F,A}`; the separate `PEMAFCa`/`PEMAFCf` sketches
   under `Documents/Arduino/` are stale/orphaned (reference classes that were
   merged away) and won't compile against the current library. "Both F and A
   types" means both modes of the one algorithm.
2. **Only one real measurement exists on disk** (`analysis/data/self/AUDIO001_ir_hp100.npz`,
   1225 individual per-period IRs from one physical session). You've
   confirmed: use individual per-period IRs from this one recording as a
   pipeline-validation stand-in for "multiple conditions" for now (train on
   some, hold out others), understanding this exercises the framework
   end-to-end without yet being a real test of PCA's cross-condition benefit.
   The framework must work identically once real multi-condition `.npz`
   files exist — nothing should hardcode the single-recording case.
3. **A real correctness bug, found while verifying the design, in the
   *existing, shipped* `train_pca_feedback_model.py`** (not just relevant to
   this new work): `window_around_peak()` anchors each measurement's window
   at `peak - pre_peak` (default `pre_peak=8`), so the trained model's
   content always sits near tap index 8 regardless of the impulse's true
   absolute delay. But the live device's `efbp`/ring-buffer indexing is
   zero-lag-anchored (tap 0 = "now", no re-anchoring) with `delay_samples=0`
   (nonzero delay is explicitly flagged buggy by the library author). Our
   one real measurement peaks at ~7.02 ms (~337 samples @ 48kHz) — with the
   default `pre_peak=8`, a model trained via the documented README workflow
   would place its energy ~329 samples away from where the real ring buffer
   would look for it. This isn't just a simulator concern: if this pipeline
   is used to train a model for actual on-device deployment with the
   documented default, the resulting `pca_model.csv` would likely be
   badly misaligned. This plan works around it for the simulator (see
   §4) and separately flags it for you to decide whether/how to fix in
   `train_pca_feedback_model.py` itself — out of scope to silently change
   here since it's previously-documented, previously-shipped behavior.
4. **Scope decision: open-loop simulation, stated explicitly.** The
   loudspeaker/forward signal is an independent, known excitation (noise, or
   later a WAV file) convolved against the *true* feedback-path IR to
   produce the contaminated mic signal — not a full closed hearing-aid loop
   where the AFC's own output (through gain/limiter/hardclip) is what
   actually gets played and re-coupled. This matches how AFC
   misalignment/ERLE convergence is conventionally evaluated in the
   literature (the Spriet et al. reference the sketches themselves cite),
   matches what you asked for (compare algorithms' tracking/convergence as
   the path changes, not closed-loop howling/stability), and is what makes
   the simulation fast (the per-epoch contaminated mic signal can be
   precomputed once via FFT convolution and shared across all algorithm
   variants, rather than needing a true sample-by-sample circular solve).
   Closed-loop/stability testing would be a natural, separate future
   extension given the stated goal of adding more evaluations later.

## Package layout

New package `analysis/afc_sim/`:

```
afc_sim/
  algorithms/
    base.py         # AFCAlgorithm ABC: reset(), process_sample(mic, spk)->e,
                     #   process_block(...) default+overridable fast path,
                     #   get_estimated_ir()->ndarray, name, afl properties
    registry.py      # register_algorithm()/create_algorithm() by name string
    whitening.py      # ARWhitener: owns the raw reference/mic ring buffers,
                       #   mode F (fixed alpha) / mode A (Levinson-Durbin,
                       #   re-estimated every ar_block_len samples of e[i]),
                       #   returns both raw and whitened windows per sample
    pemafc.py          # PEMAFCFull: raw fb_est/e[i] via dot(offset_u, efbp);
                        #   NLMS update on whitened (u_f, e_f). Composes an
                        #   ARWhitener (not a subclass relationship).
    pemafc_pca.py        # PEMAFCPCA: efbp = mean_ir + sum(c[k]*component[k]);
                          #   update in the K-dim projected space. Also
                          #   composes an ARWhitener. Loads a pca_model.csv.
  feedback_path.py         # FeedbackPath: holds the truncated ground-truth
                            #   IR, precomputes contaminated mic signal for
                            #   an epoch via scipy.signal.fftconvolve,
                            #   set_ir() to switch between epochs
  conditions.py              # loads the .npz, computes a per-period local
                              #   SNR/quality score (peak vs local-noise-floor
                              #   ratio, reusing the approach from earlier in
                              #   this conversation), splits into disjoint
                              #   train/held-out period-index sets
  pca_training.py             # thin wrapper: builds zero-lag-truncated
                               #   (raw_ir[:afl], NOT peak-anchored) windows
                               #   from selected training conditions, calls
                               #   the refactored fit_pca()/write_pca_csv()
  signals.py                   # excitation generators: seeded white/pink
                                #   noise; load-from-wav
  metrics.py                    # misalignment/NPM (dB) against raw_ir[:afl]
                                 #   zero-lag ground truth (NOT peak-anchored
                                 #   -- see §4); ERLE via sliding-window power
                                 #   ratio; convergence-time-to-threshold
                                 #   after each switch
  simulate.py                    # SimulationConfig/EpochSpec/TrialResult
                                  #   dataclasses; run_simulation() drives
                                  #   one shared excitation realization per
                                  #   epoch through each algorithm variant
  results_io.py                  # save metrics/config/manifest, comparison
                                  #   plots (misalignment & ERLE vs. time,
                                  #   switch points marked)
  run_experiment.py               # CLI entrypoint wiring everything together
```

`analysis/train_pca_feedback_model.py`: minimal refactor only — extract the
existing inline SVD block (lines ~158-194) into `fit_pca(windows,
n_components=None, var_explained=0.95, device_max_components=16) ->
(mean_ir, components, ratio, cumulative, K)` and the CSV-writing block into
`write_pca_csv(path, afl, mean_ir, components)`, with `main()` calling both.
Output/CLI behavior unchanged byte-for-byte; this just makes the math
importable so `pca_training.py` reuses it instead of duplicating it.

## Algorithm port — the parts most likely to be silently wrong

- **Per-sample pipeline order** (mode A's whitening depends on the error
  signal the update itself produces, so order matters): (1) compute raw
  `fb_est = dot(offset_u, efbp)`, `e[i] = mic[i]-fb_est`, using the *current*
  `efbp`; (2) push `e[i]` into the AR-block accumulator; (3) whiten
  mic/reference with the *current* `ar_coeffs` -> `y_f`, `u_f`; (4) NLMS/PCA
  update from `y_f`, `u_f`; (5) once `ar_block_len` error-samples have
  accumulated (mode A only), run Levinson-Durbin on the block's **biased**
  autocorrelation (divide by N at every lag — required for the
  positive-semidefinite guarantee Levinson-Durbin needs), with the
  `|reflection coeff| >= 0.999` guard aborting the *whole* candidate
  coefficient set (keep old `ar_coeffs`), then reset the accumulator.
- **Ring buffers must be real preallocated circular buffers** (modulo
  indexing), not `np.roll`/list-shift per sample — at afl=512 this is the
  difference between a fast and a very slow simulation.
- **Regression test**: mode F must degenerate to `y_f[i] = mic[i] -
  alpha*mic[i-1]` exactly, i.e. match `scipy.signal.lfilter([1,-alpha],[1],mic)`.
  Cheap, high-value, verify before trusting mode A.
- **`efbp` init**: zero for the full-rank variant; `mean_ir` (from the loaded
  PCA model, `c=0`) for the PCA variant. Noted as a reasonable standard
  assumption (matches zero-initialized member-array semantics) rather than
  a byte-verified C++ fact — cheap to double check against source later if
  it matters.
- **No invented NaN/Inf recovery inside the algorithm** — the research
  didn't establish the base class has general guard-and-reset behavior
  beyond the Levinson-Durbin stability guard, so don't fabricate it. Instead
  add a simulation-level check in `simulate.py` that aborts a trial with a
  clear message if `efbp`/`e[i]` goes non-finite, so a diverging run doesn't
  silently burn compute or produce an uninterpretable plot.
- **`n_components_active` vs `n_components_loaded`**: keep as two separate,
  independently settable values on `PEMAFCPCA` so K can be swept (0..loaded)
  against one fitted model without refitting.

## Parameters (24kHz real-hardware values -> 48kHz simulation)

Real deployed values (from the actual working `.ino` sketches, 24kHz/16-sample
blocks): `mu=0.001, delta=1e-3, afl=256, alpha=0.9, ar_order=12,
ar_block_len=320, ar_reg=1e-5, delay_samples=0`.

Rescaled for the 48kHz simulation (matching the codebase's own
`*(fs/24000)` convention, applied to anything measured in samples-of-time):
- `afl = 512` (10.67ms reach either way) — cross-checked against the one
  real measurement's ~7.02ms peak delay, comfortably inside this window at
  zero lag.
- `ar_block_len = 640`
- `block_size = 32` (the one-audio-block loopback delay stays ~0.667ms of
  real time either way, not half of it — missed in the initial estimate,
  caught in design review)
- `mu, delta, alpha, ar_order, ar_reg, delay_samples` unchanged (`delay_samples`
  stays 0 — not implementing the flagged-buggy nonzero case).

Documented caveats, not fixed: `mu` unchanged means the same number of
per-sample updates happen twice as fast in wall-clock time at 48kHz vs
24kHz, so **convergence-time-in-seconds from this simulator should not be
read as a prediction of real 24kHz-hardware convergence time** — it's valid
for comparing algorithm variants against each other within the simulator.
`alpha=0.9` is carried over literally rather than fs-corrected
(`0.9^(24000/48000) ≈ 0.949` would be the fs-invariant corner-frequency
value) since it's unclear from the source whether 0.9 was tuned to a
physical property independent of fs — left as an easy config override if
you want to try the corrected value.

## Ground truth, windowing, and the pre_peak fix (this simulator only)

`FeedbackPath` convolves the excitation against a **truncated but
unwindowed** raw IR (default 2048 taps ≈ 42.7ms — comfortably longer than
the ~10ms decay already observed, short enough to keep FFT convolution
cheap, and long enough to expose a real unmodelable tail beyond afl=512,
which is the point of testing this rather than hiding it).

Both the misalignment metric's ground-truth target and `pca_training.py`'s
training windows use **zero-lag truncation** (`raw_ir[:afl]`, zero-padded if
short) — *not* `train_pca_feedback_model.py`'s peak-anchored
`window_around_peak()`. This matches how the real ring buffer actually
indexes `efbp` (tap 0 = now, no re-anchoring) and keeps the simulated ground
truth, the PCA training input, and the metric all on the same consistent
basis. `conditions.py` and `pca_training.py` do not import
`window_around_peak` for this reason.

## Simulation loop

- `conditions.py` scores each of the 1225 periods by local SNR, picks a
  disjoint train-set (default: 40 periods) and held-out eval-set (default: 6
  periods) from the higher-quality periods, fixed seed for reproducibility.
- `pca_training.py` fits a PCA model from the train-set periods only.
- `simulate.py` builds a schedule of epochs (default: each held-out period,
  20s each, configurable), generates one seeded noise (or WAV) excitation
  realization per epoch, and for each of `{PEMAFCFull(mode=F),
  PEMAFCFull(mode=A), PEMAFCPCA(mode=A)}` (the registry makes adding more
  variants later a one-line addition) runs that same excitation against that
  epoch's `FeedbackPath`, recording misalignment/ERLE time series.
- **Reset semantics**: `reset()` is called once per algorithm at the very
  start of the whole run, *not* between epochs, by default — this measures
  real reconvergence behavior after a path switch, which is what "let it
  converge before switching to the next impulse" is actually testing.
  `reset_between_epochs=True` is available as a config flag for the
  different (also legitimate) question of best-case per-condition
  convergence in isolation.
- Performance: the per-sample adaptive-update loop is irreducibly
  sequential, but the whitening step is not (within a mode-A block,
  `ar_coeffs` is fixed) — precompute `u_f` for a whole `ar_block_len` chunk
  at once via `scipy.signal.lfilter` against the (fully known, open-loop)
  excitation signal, and similarly batch `y_f` against the precomputed mic
  signal, leaving only the genuinely sequential `fb_est`/update as a
  per-sample Python loop. Will time a short run early and adjust default
  epoch count/duration if wall-clock time is impractical (no `numba`
  available in this environment).

## Verification

- Unit test: mode-F whitening matches `scipy.signal.lfilter([1,-alpha],[1],x)`.
- Unit test: `PEMAFCPCA` with `n_components_active=0` stays pinned exactly at
  `mean_ir` (no drift) — confirms the K=0 "disabled adaptation" edge case.
- Sanity check before trusting results: plot where `fit_pca()`'s output
  peaks vs. where `raw_ir[:afl]` peaks for the one real measurement, to
  confirm the zero-lag-windowing fix actually produces an aligned model
  (this is the concrete, cheap check the design review recommended before
  trusting any PCA-vs-full-rank comparison).
- End-to-end run: execute `run_experiment.py` with default config, confirm
  it completes in a practical wall-clock time, and inspect the
  misalignment/ERLE-vs-time plot for a visibly-worse-then-recovering
  transient at each epoch switch for all three variants — a run that shows
  no reaction at switch points would indicate the switching mechanism itself
  is broken, independent of any algorithm's actual quality.
- Explicit caveat to carry into any interpretation of results: with only one
  physical recording session behind the "conditions," this first run
  validates that the framework works, not that PCA-constraining helps in
  reality — that claim needs real distinct-condition recordings.
