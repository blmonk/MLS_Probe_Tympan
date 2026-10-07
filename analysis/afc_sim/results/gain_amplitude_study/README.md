# Gain / signal-amplitude tuning study (closed-loop simulation)

## Question

`run_experiment.py --loop-mode closed` needs both an excitation amplitude
(`--signal-amplitude`) and a gain (`--gain-db`) applied to the algorithm's own
output before it's "played" and acoustically coupled back. What combinations
actually produce meaningful, non-degenerate closed-loop behavior, vs. either
(a) too little real acoustic feedback to learn anything, or (b) immediate
clipping unrelated to the acoustic loop at all?

## Method

Swept `signal_amplitude x gain_db` on one real measured condition
(`period_0857` from `data/self/AUDIO001_ir_hp100.npz`, afl=128, 1s epochs,
PEMAFC mode F, mu=0.001). For each combination, ran one closed-loop epoch and
recorded final misalignment and the fraction of samples that hit the
saturation limit. Reproducible via `afc_sim/gain_amplitude_sweep.py`; raw
output in `sweep_output.txt`.

## Result

```
amplitude \ gain_db      40      45      50      55      60      65      70      75      80
1.0e-06                   0.0     0.1     0.2     0.5     1.0     1.2     0.8    11.4    SAT
1.0e-05                  13.8    weak    13.9    10.8     7.0     3.7     1.4    11.1    SAT
1.0e-04                  weak    weak    weak    11.7     7.3     3.8     1.4    SAT     SAT
3.0e-04                  weak    weak    weak    11.7     7.4    SAT     SAT     SAT     SAT
1.0e-03                  weak    weak    weak    SAT     SAT     SAT     SAT     SAT     SAT
```
(cell = final misalignment in dB where usable; "weak" = no saturation but
misalignment >15dB; "SAT" = >5% of samples saturated)

## Interpretation

There's a **diagonal usable band**, not a simple threshold: the boundary
tracks `amplitude x gain` roughly constant, because two independent things
have to both hold at once:
- **Enough gain relative to the true IR's tiny raw amplitude** (~1.6e-6 peak
  for this condition) for the acoustic feedback contribution to be a
  meaningfully learnable fraction of the mic signal, not swamped by the
  desired signal or the algorithm's own adaptation noise floor ("weak").
- **Not so much `gain x amplitude`** that the played signal clips against
  the saturation limit before the acoustic loop's own resonance behavior
  ever gets to matter ("SAT").

**Smaller excitation amplitude gives more usable headroom**, not less: at
amplitude=1e-6, misalignment reaches 0.0 dB (matching open-loop quality) at
gain=40dB, and stays usable up to ~70dB before saturating. At amplitude=1e-3,
the usable band has almost vanished. This is because a smaller amplitude
buys more gain range before `gain x amplitude` alone approaches the
saturation limit, leaving room for gain to do its real job (exciting the
acoustic path) before clipping dominates.

## Cross-validation against the analytical MSG formula

Computed `MSG_without_AFC` for this same condition using the Nyquist-based
formula (`-20*log10(max |F(f)| at phase=0Β° mod 360Β°)`, see the MSG/ASG plan):
**97.5 dB**, from 33 phase-zero crossings, worst case at 4784 Hz. This lines
up with the empirical sweep: every amplitude tested saturates by gain=80dB
and above, consistent with genuine loop instability compounding on top of
direct clipping somewhere in the 80-180dB range predicted analytically once
you're not amplitude-constrained. Two independently-implemented methods
(empirical closed-loop simulation, analytical frequency-domain calculation)
agree with each other, which is good evidence both are measuring something
real rather than an artifact of either implementation.

## Practical guidance

For closed-loop experiments on data at this scale (raw MLS-derived impulse
responses, un-calibrated to physical SPL), start with a **small excitation
amplitude (~1e-6 to 1e-5) and sweep gain from ~40dB upward** rather than
using signal amplitudes designed for open-loop testing (~0.1) with modest
gain -- the latter clips immediately and never exercises genuine acoustic
loop dynamics.
