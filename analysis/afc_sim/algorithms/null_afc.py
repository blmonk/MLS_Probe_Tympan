"""
NullAFC: no-cancellation baseline. efbp is always zero; e == mic always.

Two purposes: (1) a legitimate "how much do any of these algorithms help at
all" baseline for misalignment/ERLE/ASG comparisons; (2) the structurally
correct topology for empirically validating MSG_without_AFC via the
closed-loop simulator -- zero cancellation is exactly the textbook
single-loop Nyquist scenario the MSG formula models, cleaner than
validating against a partially-converged adaptive algorithm whose own
cancellation would confound the comparison.
"""

import numpy as np

from .base import AFCAlgorithm
from .registry import register_algorithm


class NullAFC(AFCAlgorithm):
    def __init__(self, afl: int):
        self._afl = afl

    def reset(self) -> None:
        pass

    def process_block(self, mic: np.ndarray, spk: np.ndarray) -> np.ndarray:
        return np.asarray(mic, dtype=np.float64).copy()

    def push_reference_sample(self, true_spk_sample: float) -> None:
        pass  # no regressor history to maintain -- never adapts

    def get_estimated_ir(self) -> np.ndarray:
        return np.zeros(self._afl)

    @property
    def name(self) -> str:
        return "NullAFC"

    @property
    def afl(self) -> int:
        return self._afl


@register_algorithm("null_afc")
def _make_null_afc(**kwargs) -> NullAFC:
    # Accept and ignore any mu/delta/alpha/etc. passed via common_kwargs so
    # NullAFC can be selected via --variants alongside the real algorithms
    # without special-casing the CLI's argument wiring.
    return NullAFC(afl=kwargs["afl"])
