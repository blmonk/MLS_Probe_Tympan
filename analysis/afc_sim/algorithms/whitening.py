"""
Prediction-error (AR) whitening filter shared by PEMAFCFull and PEMAFCPCA,
faithfully porting AudioFeedbackCancelPEMAFC_F32's whitening math.

Whitening convention (verified against Tympan_Library source):
    y_f[n] = y[n] + sum_{j=0}^{p-1} ar_coeffs[j] * y[n-1-j]
    u_f[k] = offset_u[k] + sum_{j=0}^{p-1} ar_coeffs[j] * offset_u[k+j+1]

Mode F: p=1 fixed, ar_coeffs=[-alpha] (standard first-order pre-emphasis).
Mode A: p=ar_order, ar_coeffs re-estimated every ar_block_len samples of the
        AFC's own error signal e[i] via Levinson-Durbin on the biased
        autocorrelation, with a |reflection coeff| >= 0.999 stability guard
        that discards the whole candidate update and keeps the old coeffs.

Both y_f[n] and u_f[k] are, for FIXED ar_coeffs, exactly a pure-FIR filter
`[1, *ar_coeffs]` applied to the raw signal (y or the loudspeaker reference
u respectively) -- see PEMAFCMode/whiten_fir below. Since this simulator is
open-loop (the full reference and, given a fixed active impulse response,
the full contaminated mic signal are known in advance for an epoch), this
lets whitening be computed via scipy.signal.lfilter in one call per
AR-stable chunk instead of a literal per-sample ring buffer -- same math,
much less per-sample Python overhead.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import lfilter


class PEMAFCMode(Enum):
    F = "F"
    A = "A"


def biased_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """r[k] = sum_{n=k}^{N-1} x[n]*x[n-k] / N for k=0..max_lag (biased -- always
    divide by N, not N-k -- required for the Toeplitz matrix Levinson-Durbin
    needs to stay positive-semidefinite)."""
    N = len(x)
    r = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        r[k] = np.dot(x[k:], x[: N - k]) / N if N > k else 0.0
    return r


def levinson_durbin(r: np.ndarray, order: int):
    """
    Levinson-Durbin recursion on autocorrelation r[0..order]. Returns
    (coeffs, stable): coeffs has length `order`, coeffs[j-1] == a[j] in the
    standard A(z) = 1 + sum a[j] z^-j convention, which is exactly the
    ar_coeffs[] used by whiten_fir below. `stable=False` (coeffs all zero)
    if |reflection coeff| >= 0.999 or non-finite at any recursion order --
    caller must discard and keep the previous ar_coeffs in that case.
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if not np.isfinite(e) or e <= 1e-12:
        return np.zeros(order), False
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[i - 1 : 0 : -1]))
        if not np.isfinite(e) or e <= 1e-12:
            return np.zeros(order), False
        k = -acc / e
        if not np.isfinite(k) or abs(k) >= 0.999:
            return np.zeros(order), False
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        a[i] = k
        e *= 1.0 - k * k
    return a[1:].copy(), True


def whiten_fir(x: np.ndarray, ar_coeffs: np.ndarray, lookback: np.ndarray) -> np.ndarray:
    """
    Apply y_f[n] = x[n] + sum_j ar_coeffs[j]*x[n-1-j] to `x`, using `lookback`
    (the p most recent samples immediately preceding x[0], oldest-first, zero
    -padded at the very start of a signal) so the result is continuous across
    chunk boundaries without needing scipy's lfilter `zi` state machinery.
    """
    p = len(ar_coeffs)
    if p == 0:
        return x.copy()
    extended = np.concatenate([lookback[-p:], x])
    b = np.concatenate([[1.0], ar_coeffs])
    filtered = lfilter(b, [1.0], extended)
    return filtered[p:]


@dataclass
class ARWhitenerConfig:
    mode: PEMAFCMode
    ar_order: int = 12
    alpha: float = 0.9
    ar_block_len: int = 640
    ar_reg: float = 1e-5


class ARWhitener:
    """
    Owns ar_coeffs and (mode A) the running error-signal accumulator used to
    re-estimate them. Stateless with respect to the raw reference/mic
    signals themselves -- callers pass full arrays per chunk (see
    algorithms/pemafc.py), since this simulator precomputes those arrays
    open-loop rather than streaming them through a ring buffer.
    """

    def __init__(self, cfg: ARWhitenerConfig):
        self.cfg = cfg
        self.p = 1 if cfg.mode == PEMAFCMode.F else cfg.ar_order
        self.reset()

    def reset(self) -> None:
        if self.cfg.mode == PEMAFCMode.F:
            self.ar_coeffs = np.array([-self.cfg.alpha])
        else:
            self.ar_coeffs = np.zeros(self.cfg.ar_order)
        self._err_accum: list[float] = []

    def whiten_chunk(self, spk: np.ndarray, mic: np.ndarray,
                      spk_lookback: np.ndarray, mic_lookback: np.ndarray):
        """Whiten one AR-stable chunk (fixed ar_coeffs throughout). Returns
        (u_f_full, y_f_full), each the same length as spk/mic."""
        u_f = whiten_fir(spk, self.ar_coeffs, spk_lookback)
        y_f = whiten_fir(mic, self.ar_coeffs, mic_lookback)
        return u_f, y_f

    def push_errors(self, e_chunk: np.ndarray) -> None:
        """Mode A only: accumulate error samples; re-estimate ar_coeffs via
        Levinson-Durbin whenever ar_block_len samples have accumulated
        (possibly more than once if e_chunk spans multiple blocks)."""
        if self.cfg.mode != PEMAFCMode.A:
            return
        self._err_accum.extend(e_chunk.tolist())
        while len(self._err_accum) >= self.cfg.ar_block_len:
            block = np.array(self._err_accum[: self.cfg.ar_block_len])
            self._err_accum = self._err_accum[self.cfg.ar_block_len :]
            block = block - block.mean()
            r = biased_autocorr(block, self.cfg.ar_order)
            r[0] += self.cfg.ar_reg * r[0] + self.cfg.ar_reg
            coeffs, stable = levinson_durbin(r, self.cfg.ar_order)
            if stable:
                self.ar_coeffs = coeffs
            # else: keep previous ar_coeffs unchanged, matching the C++ guard.

    def next_reestimate_in(self) -> int:
        """Mode A: samples of error still needed before the next re-estimation
        (used by pemafc.py to size AR-stable sub-chunks). Mode F: a large
        sentinel since ar_coeffs never changes."""
        if self.cfg.mode != PEMAFCMode.A:
            return 1 << 30
        return self.cfg.ar_block_len - len(self._err_accum)
