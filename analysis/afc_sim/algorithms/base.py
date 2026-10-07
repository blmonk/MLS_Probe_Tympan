"""AFCAlgorithm: the common interface every cancellation algorithm variant
implements, so simulate.py and future additions (beyond PEMAFC/PCA) can be
driven identically."""

from abc import ABC, abstractmethod

import numpy as np


class AFCAlgorithm(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset all adaptive state (efbp/c[], whitening coefficients/accumulator)."""

    @abstractmethod
    def process_block(self, mic: np.ndarray, spk: np.ndarray) -> np.ndarray:
        """
        Process a block of samples. `mic` is the feedback-contaminated
        microphone signal (from FeedbackPath), `spk` is the loudspeaker
        reference signal for the same samples. Returns e[] (mic minus the
        algorithm's raw, un-whitened feedback estimate) -- same length as
        the inputs. May be called with any block length, including 1;
        internal AR re-estimation chunking (mode A) is handled independent
        of the caller's block size.
        """

    def process_sample(self, mic_sample: float, spk_sample: float) -> float:
        """Convenience single-sample wrapper around process_block."""
        return float(self.process_block(np.array([mic_sample]), np.array([spk_sample]))[0])

    def push_reference_sample(self, true_spk_sample: float) -> None:
        """
        Closed-loop only. The regressor's `spk` tap-0 is "one sample ago" --
        the minimum any feedback loop needs so a sample doesn't depend on
        itself (see algorithms/pemafc.py's module docstring). In closed
        loop, that one-sample-ago value is only known *after*
        process_block() has already returned e[i] for the current sample
        (since played[i] = gain*e[i]) -- process_block()'s own `spk`
        argument is therefore necessarily a step too late to carry it.
        Call this immediately after computing played[i] from e[i] to record
        the real value for use as tap-0 when processing sample i+1; pass
        anything (e.g. 0.0) as process_block()'s own `spk` argument in
        closed loop, since it's provably unused for the current sample's
        own output (the same-call window never reaches the newly-appended
        value -- see pemafc.py/_process_chunk).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support closed-loop use")

    @abstractmethod
    def get_estimated_ir(self) -> np.ndarray:
        """Current afl-length feedback-path estimate (efbp)."""

    def get_diagnostics(self) -> dict:
        """Optional scalar diagnostics beyond efbp itself, for algorithms with
        other interesting time-varying internal state (e.g. a dynamically
        adapted mixing weight) -- empty dict by default. Snapshotted by
        simulate.py at the same cadence as misalignment/ERLE; plotted only
        for whichever algorithms actually report something."""
        return {}

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def afl(self) -> int: ...
