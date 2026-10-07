"""
FeedbackPath: convolves a known excitation signal against the active
ground-truth impulse response to produce the feedback-contaminated mic
signal for one epoch (open-loop -- see afc_sim_PLAN.md Β§ "Scope decision").

Uses a truncated-but-unwindowed raw IR (not the afl-length,
peak-anchored window used for PCA training/metrics) so the simulated
ground truth retains a realistic tail beyond what any afl-tap adaptive
filter can represent -- that unmodelable tail is part of what's worth
testing, not something to hide.

This class only owns "which IR is active and how to convolve it" -- it does
NOT own the epoch-switching schedule (that's simulate.py's job, so the same
schedule stays identical across every algorithm variant being compared).
"""

import numpy as np
from scipy.signal import fftconvolve


class FeedbackPath:
    def __init__(self, ir_truncate_len: int = 2048):
        self.ir_truncate_len = ir_truncate_len
        self._ir = None

    def set_ir(self, raw_ir: np.ndarray) -> None:
        """raw_ir: the full measured impulse response (e.g. mls_length samples).
        Truncated (zero-padded if shorter) to ir_truncate_len."""
        ir = np.asarray(raw_ir, dtype=np.float64)
        L = self.ir_truncate_len
        if len(ir) >= L:
            self._ir = ir[:L].copy()
        else:
            self._ir = np.concatenate([ir, np.zeros(L - len(ir))])

    @property
    def active_ir(self) -> np.ndarray:
        if self._ir is None:
            raise RuntimeError("FeedbackPath.set_ir() has not been called yet")
        return self._ir

    def contaminate(self, spk: np.ndarray) -> np.ndarray:
        """mic[n] = sum_k spk[n-k]*ir[k] (causal, zero history before spk[0])."""
        if self._ir is None:
            raise RuntimeError("FeedbackPath.set_ir() has not been called yet")
        full = fftconvolve(spk, self._ir, mode="full")
        return full[: len(spk)]
