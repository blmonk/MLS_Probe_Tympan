"""
Metrics for comparing AFC algorithm variants: misalignment/NPM (dB) against
a zero-lag-truncated ground truth (NOT peak-anchored -- see
afc_sim_PLAN.md Β§4, this must match conditions.zero_lag_window's convention
since that's how efbp is actually indexed), ERLE via block power ratios,
convergence-time-to-threshold after a switch, and MSG/ASG (Maximum/Added
Stable Gain) via the standard Nyquist-criterion frequency-domain formula.
"""

import numpy as np


def misalignment_db(estimated_ir: np.ndarray, true_ir_afl: np.ndarray) -> float:
    """20*log10(||true - est|| / ||true||) -- standard AFC misalignment (NPM),
    lower is better, 0 dB would be a perfect estimate. true_ir_afl must
    already be the same afl-length, zero-lag-truncated ground truth used to
    build the FeedbackPath/PCA training for this condition."""
    num = np.linalg.norm(true_ir_afl - estimated_ir)
    den = np.linalg.norm(true_ir_afl) + 1e-15
    return 20.0 * np.log10(num / den + 1e-15)


def erle_db(mic_block: np.ndarray, e_block: np.ndarray) -> float:
    """10*log10(E[mic^2]/E[e^2]) over one block -- higher is better."""
    mic_pow = np.mean(mic_block**2) + 1e-20
    e_pow = np.mean(e_block**2) + 1e-20
    return 10.0 * np.log10(mic_pow / e_pow)


def convergence_time_s(times_s: np.ndarray, misalignment_series_db: np.ndarray,
                        threshold_db: float, switch_time_s: float = 0.0):
    """First time (seconds, relative to the start of the array) that
    misalignment drops to/below threshold_db, restricted to samples at or
    after switch_time_s. Returns None if never reached within the series."""
    mask = times_s >= switch_time_s
    idx = np.where(mask & (misalignment_series_db <= threshold_db))[0]
    if len(idx) == 0:
        return None
    return float(times_s[idx[0]])


# ── MSG / ASG (Maximum / Added Stable Gain) ─────────────────────────────────
#
# Standard Nyquist-criterion formula from the hearing-aid AFC literature: for
# a feedback path with frequency response F(f) and a real, positive,
# frequency-flat scalar forward gain G, the loop is unstable if some
# frequency has |G*F(f)| >= 1 AND phase(F(f)) ~= 0 deg (mod 360 deg) --
# reinforcement. So:
#
#   MSG_dB(ir) = -20*log10( max over {f: Im(F(f))=0, Re(F(f))>0} of |F(f)| )
#
# MSG_without_AFC = MSG_dB(true_ir_afl); MSG_with_AFC(t) = MSG_dB(true_ir_afl
# - efbp(t)) (the residual/uncancelled path); ASG(t) = the difference.
#
# NOTE: this measures stability margin considering only the first `afl` taps
# of the feedback path (since efbp is only afl taps long), not the full
# ir_truncate_len acoustic path feedback_path.py otherwise preserves. Close
# in practice for the one real condition cross-validated so far (see
# afc_sim_PLAN.md), not a structural guarantee -- see the fft_len vs.
# ir_truncate_len sanity check in the test suite.

def next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def default_msg_fft_len(afl: int, oversample: int = 32) -> int:
    """Default fft_len for msg_db(), derived from afl rather than hardcoded
    -- a fixed fft_len validated at one afl silently under-resolves at a
    larger one. oversample=32 is comfortably above the 64x ratio this was
    validated at for afl=128 (8192/128)."""
    return next_pow2(oversample * afl)


def frequency_response(ir: np.ndarray, fft_len: int, fs: float = 1.0):
    """rfft(ir, n=fft_len) plus matching frequency axis (Hz if fs given,
    else cycles/sample). fft_len < len(ir) would silently *truncate* ir in
    the time domain (not zero-pad) -- a real correctness bug, not a
    leakage concern (ir is the complete signal of interest by construction,
    so any fft_len >= len(ir) samples its exact DTFT with no approximation
    error, just at whatever density fft_len gives)."""
    assert fft_len >= len(ir), (
        f"fft_len={fft_len} < len(ir)={len(ir)} would truncate ir, not zero-pad it")
    F = np.fft.rfft(ir, n=fft_len)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / fs)
    return freqs, F


def find_zero_phase_crossings(freqs: np.ndarray, F: np.ndarray):
    """
    Returns a list of (freq, magnitude) for every frequency where Im(F)=0
    and Re(F)>0 (phase = 0 deg mod 360 deg -- a candidate for sustained
    in-phase reinforcement around a feedback loop).

    Two independent passes, not one combined sign-diff scan, to avoid both
    a double-count and a missed-crossing failure mode on exact zeros:
      1. Bins where Im(F[k]) == 0 exactly -- added directly, no
         interpolation (none needed/possible). This explicitly includes the
         DC bin (k=0) and, when fft_len is even, the Nyquist bin -- both
         structurally have Im=0 for any real-valued input, not a
         data-dependent edge case, and would be missed by an interior-only
         scan.
      2. Strict opposite-sign adjacent bins (Im[k]*Im[k+1] < 0) -- genuine
         interior crossings, found by linearly interpolating Re and Im
         *separately* to the sub-bin crossing point (not interpolating |F|
         directly, which isn't equivalent), then testing Re>0 there.
    """
    re = F.real
    im = F.imag
    n = len(F)
    crossings = []

    for k in range(n):
        if im[k] == 0.0 and re[k] > 0:
            crossings.append((float(freqs[k]), float(abs(re[k]))))

    for k in range(n - 1):
        if im[k] == 0.0 or im[k + 1] == 0.0:
            continue  # exact zeros already handled above
        if im[k] * im[k + 1] < 0:
            t = im[k] / (im[k] - im[k + 1])
            re_c = re[k] + t * (re[k + 1] - re[k])
            if re_c > 0:
                f_c = freqs[k] + t * (freqs[k + 1] - freqs[k])
                crossings.append((float(f_c), float(abs(re_c))))

    return crossings


def msg_db(ir: np.ndarray, fft_len: int, fs: float = 1.0):
    """MSG in dB for the given impulse response. Returns +inf if no
    zero-phase, positive-real-part crossing exists (can happen as a
    residual -> 0, e.g. efbp converges toward true_ir_afl exactly) --
    semantically correct: no residual feedback path found means
    unconditionally stable, not an error."""
    freqs, F = frequency_response(ir, fft_len, fs)
    crossings = find_zero_phase_crossings(freqs, F)
    if not crossings:
        return float("inf")
    worst_mag = max(mag for _, mag in crossings)
    return -20.0 * np.log10(worst_mag + 1e-300)


def asg_db(msg_with_afc_db: float, msg_without_afc_db: float) -> float:
    """Plain subtraction -- takes already-computed MSG values, does not
    itself touch the FFT. Keeping this a pure subtraction is what lets
    MSG_without_AFC be computed once per epoch (constant) rather than
    recomputed on every snapshot."""
    return msg_with_afc_db - msg_without_afc_db
