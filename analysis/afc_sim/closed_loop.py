"""
Closed-loop epoch simulation: the algorithm's own output (after gain and
saturation) is what actually gets "played" and acoustically coupled back
through the true impulse response into future mic samples -- unlike
feedback_path.py's open-loop convolution (an independent, pre-known
excitation), this is a genuine feedback loop and can genuinely diverge
("howl") if gain is high enough.

Causality: mic[i] depends on played samples through i-1 only (the minimum
any feedback loop needs -- a sample can't depend on itself):

  mic[i] = desired[i] + sum_j true_ir[j] * played[i-1-j]

Processing is genuinely sample-by-sample: played[i] = gain*e[i] is only
known after the algorithm has produced e[i] for this sample, so it's pushed
into the algorithm's regressor history via push_reference_sample()
immediately afterward, for use starting at sample i+1 -- see
AFCAlgorithm.push_reference_sample()'s docstring for why process_block()'s
own `spk` argument can't carry it directly. This earlier simplified away a
block_delay-sized margin that existed only to let the old version batch
multiple samples per call (matching real Tympan hardware's one-audio-block
loopback latency); not needed for an offline simulator, and the per-sample
loop here is simpler to read and trust even though it's slower.
"""

import numpy as np

from .algorithms.base import AFCAlgorithm


def run_closed_loop_epoch(algo: AFCAlgorithm, desired: np.ndarray, true_ir: np.ndarray,
                           gain_linear: float, saturation_limit: float = 1.0, on_chunk=None):
    """
    Returns (mic, played, e, saturated_fraction).

    on_chunk(pos, mic_chunk, e_chunk), if given, is called after every
    sample (chunk of size 1) -- simulate.py uses this to snapshot
    misalignment/ERLE at the same per-sample granularity the open-loop path
    can offer, just driven from here since this is the loop with access to
    algo's live state as it goes.
    """
    n = len(desired)
    ir_len = len(true_ir)
    pad = ir_len
    played_full = np.zeros(pad + n)  # played_full[pad+i] == played[i] once known
    mic = np.empty(n)
    e_out = np.empty(n)
    saturated_samples = 0

    for i in range(n):
        idx = pad + i
        window = played_full[idx - ir_len: idx][::-1]  # played[i-1], played[i-2], ..., played[i-ir_len]
        mic[i] = desired[i] + float(np.dot(true_ir, window))

        e_i = algo.process_block(np.array([mic[i]]), np.array([0.0]))[0]
        if not np.isfinite(e_i) or not np.all(np.isfinite(algo.get_estimated_ir())):
            raise RuntimeError(f"{algo.name} diverged (non-finite state) in closed loop at sample {i}.")

        played_i = float(np.clip(gain_linear * e_i, -saturation_limit, saturation_limit))
        if abs(played_i) >= saturation_limit * 0.999:
            saturated_samples += 1
        algo.push_reference_sample(played_i)

        played_full[idx] = played_i
        e_out[i] = e_i
        if on_chunk is not None:
            on_chunk(i + 1, mic[i:i + 1], e_out[i:i + 1])

    played = played_full[pad:]
    return mic, played, e_out, saturated_samples / n
