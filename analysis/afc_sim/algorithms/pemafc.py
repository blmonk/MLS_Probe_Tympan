"""
PEMAFCFull: full-rank port of AudioFeedbackCancelPEMAFC_F32::cha_afc(), both
mode F and mode A (see whitening.py for the shared AR-whitening front end).

Per-sample algorithm (order matters -- see the plan doc / module docstring
in whitening.py for why):
  1. fb_est = dot(u_raw, efbp); e = mic - fb_est          (current efbp)
  2. accumulate e into the AR error-block (mode A)
  3. y_f, u_f = whiten(mic, u_raw) using CURRENT ar_coeffs
  4. fb_est_f = dot(u_f, efbp); e_f = y_f - fb_est_f
  5. norm_sq = dot(u_f,u_f)+delta; efbp += (mu/norm_sq)*e_f*u_f
  6. once ar_block_len error samples accumulated (mode A): re-estimate
     ar_coeffs via Levinson-Durbin (may fail -> keep old coeffs)

Regressor convention: offset_u[0] = spk[i-1] (one sample ago), reaching back
to spk[i-afl]. This fixed 1-sample lag is the ordinary minimum causality
requirement for any feedback loop (a sample can't depend on itself) -- not a
tunable "block_delay" matching real Tympan hardware's audio-block buffering,
which this offline simulator has no reason to replicate (see afc_sim_PLAN.md
Phase 0). conditions.zero_lag_window() is shifted by the same 1 sample so
the ground truth efbp is compared against uses a matching convention.

Implementation note: this simulator is open-loop, so an entire epoch's
loudspeaker reference and (for a fixed active impulse response) mic signal
are known in advance. That lets whitening be computed via
scipy.signal.lfilter over an AR-stable sub-chunk at a time rather than a
literal per-sample ring buffer -- see whiten_fir()/whiten_chunk(). Only the
raw fb_est/update step is genuinely sequential (efbp changes every sample),
so that part remains a per-sample Python loop; everything else is batched.
"""

import numpy as np
from scipy.signal import lfilter

from .base import AFCAlgorithm
from .registry import register_algorithm
from .whitening import ARWhitener, ARWhitenerConfig, PEMAFCMode


class PEMAFCFull(AFCAlgorithm):
    def __init__(self, afl: int, mu: float, delta: float, mode: PEMAFCMode,
                 alpha: float = 0.9, ar_order: int = 12, ar_block_len: int = 640,
                 ar_reg: float = 1e-5, name_override: str | None = None):
        self._afl = afl
        self.mu = mu
        self.delta = delta
        self.whitener = ARWhitener(ARWhitenerConfig(
            mode=mode, ar_order=ar_order, alpha=alpha,
            ar_block_len=ar_block_len, ar_reg=ar_reg))
        self._mode = mode
        self._name_override = name_override
        self._hist_len = afl + max(ar_order, 1) + 8  # margin
        self.reset()

    def reset(self) -> None:
        self.efbp = np.zeros(self._afl)
        self.whitener.reset()
        self._spk_hist = np.zeros(self._hist_len)
        self._mic_hist = np.zeros(self._hist_len)

    def get_estimated_ir(self) -> np.ndarray:
        return self.efbp.copy()

    def push_reference_sample(self, true_spk_sample: float) -> None:
        self._spk_hist[-1] = true_spk_sample

    @property
    def name(self) -> str:
        return self._name_override or f"PEMAFC-{self._mode.value}"

    @property
    def afl(self) -> int:
        return self._afl

    def process_block(self, mic: np.ndarray, spk: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float64)
        spk = np.asarray(spk, dtype=np.float64)
        n = len(mic)
        e_out = np.empty(n)

        pos = 0
        while pos < n:
            chunk_len = min(n - pos, max(1, self.whitener.next_reestimate_in()))
            mic_chunk = mic[pos:pos + chunk_len]
            spk_chunk = spk[pos:pos + chunk_len]
            e_out[pos:pos + chunk_len] = self._process_chunk(mic_chunk, spk_chunk)
            # carry history forward after EVERY sub-chunk (not just at the end
            # of this call) -- a re-estimation boundary can fall mid-call,
            # splitting it into multiple sub-chunks, and a later sub-chunk
            # must see the samples the earlier one(s) just played.
            self._spk_hist = np.concatenate([self._spk_hist, spk_chunk])[-self._hist_len:]
            self._mic_hist = np.concatenate([self._mic_hist, mic_chunk])[-self._hist_len:]
            pos += chunk_len

        return e_out

    def _process_chunk(self, mic_chunk: np.ndarray, spk_chunk: np.ndarray) -> np.ndarray:
        # H (hist_len) is sized so that H > p (whitening filter order) for
        # every valid mode/ar_order, so filtering `extended` fresh with zero
        # initial conditions is exact everywhere we actually slice a window
        # from it: the only corrupted output samples (indices 0..p-1, a
        # zero-initial-condition transient) fall inside self._*_hist, well
        # before the earliest window start (see reset()'s _hist_len margin).
        H = self._hist_len
        L = len(mic_chunk)
        extended_spk = np.concatenate([self._spk_hist, spk_chunk])
        extended_mic = np.concatenate([self._mic_hist, mic_chunk])
        b = np.concatenate([[1.0], self.whitener.ar_coeffs])
        filt_spk = lfilter(b, [1.0], extended_spk)
        filt_mic = lfilter(b, [1.0], extended_mic)

        afl = self._afl
        e_chunk = np.empty(L)
        for k in range(L):
            idx = H + k  # position of this new sample within the extended arrays
            # offset_u[0] = spk[idx-1] (one sample ago) .. offset_u[afl-1] = spk[idx-afl]
            lo = idx - afl
            hi = idx
            u_raw = extended_spk[lo:hi][::-1]
            fb_est = float(np.dot(u_raw, self.efbp))
            e = mic_chunk[k] - fb_est
            e_chunk[k] = e

            u_f = filt_spk[lo:hi][::-1]
            y_f = filt_mic[idx]
            fb_est_f = float(np.dot(u_f, self.efbp))
            e_f = y_f - fb_est_f

            norm_sq = float(np.dot(u_f, u_f)) + self.delta
            mu_mult = (self.mu / norm_sq) * e_f
            self.efbp += mu_mult * u_f

        self.whitener.push_errors(e_chunk)
        return e_chunk


@register_algorithm("pemafc_f")
def _make_pemafc_f(**kwargs) -> PEMAFCFull:
    kwargs.setdefault("mode", PEMAFCMode.F)
    return PEMAFCFull(**kwargs)


@register_algorithm("pemafc_a")
def _make_pemafc_a(**kwargs) -> PEMAFCFull:
    kwargs.setdefault("mode", PEMAFCMode.A)
    return PEMAFCFull(**kwargs)


@register_algorithm("nlms")
def _make_nlms(**kwargs) -> PEMAFCFull:
    # Plain NLMS, no whitening front end at all -- mode F with alpha=0 makes
    # the whitening filter [1, -alpha] = [1, 0], an exact identity (verified
    # bit-for-bit against a hand-coded plain-NLMS reference implementation).
    # Force mode/alpha regardless of whatever was passed through common_kwargs
    # (e.g. --alpha), since "NLMS" must mean no whitening, not "whatever
    # --alpha happens to be set to".
    kwargs["mode"] = PEMAFCMode.F
    kwargs["alpha"] = 0.0
    kwargs["name_override"] = "NLMS"
    return PEMAFCFull(**kwargs)
