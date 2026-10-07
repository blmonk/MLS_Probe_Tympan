"""
PEMAFCSoftPCA: full-rank NLMS/PEMAFC adaptation (identical to PEMAFCFull) with
a "soft" PCA bias applied after every sample -- the update is NOT constrained
to the PCA subspace (unlike PEMAFCPCA, which projects the update itself), it
adapts completely freely and then gets partially pulled back toward the
mean_ir + PCA-subspace affine set every sample:

  1-4. identical whitening + raw NLMS step to PEMAFCFull -- see pemafc.py's
       module docstring for the full per-sample ordering.
  5. efbp_raw = efbp + mu_mult * u_f            (ordinary, unconstrained step)
  6. z        = efbp_raw - mean_ir
     a        = components @ z                  -- project onto the PCA basis
     z_par    = a @ components                  -- back to full afl-space
     z_perp   = z - z_par                        -- the off-subspace remainder
     efbp     = efbp_raw - soft_pca_alpha * z_perp

soft_pca_alpha in [0, 1] trades off "ordinary adaptive filter" (0) against
"hard PCA projection every sample" (1); 0 < soft_pca_alpha < 1 continuously
biases the solution toward the PCA training data without forbidding paths
outside it. Verified (see afc_sim/tests) that soft_pca_alpha=0 is bit-exact
PEMAFCFull/NLMS, as expected (z_perp is subtracted with zero weight, so
efbp is just the raw unconstrained step).

soft_pca_alpha=1 is NOT bit-exact PEMAFCPCA, despite both guaranteeing efbp
lands exactly on the mean_ir+subspace affine set after every sample -- an
earlier version of this docstring claimed otherwise and was wrong, caught by
testing rather than trusting the algebra. The two take different-sized steps
toward that subspace: PEMAFCPCA normalizes mu by the PROJECTED regressor's
energy (||components @ u_f||^2), since its NLMS update happens entirely in
the K-dim c-space; soft PCA normalizes by the FULL regressor's energy
(||u_f||^2), since its NLMS step happens in the full afl-dim space before
any projection. Those norms generically differ whenever K < afl, so the two
algorithms' trajectories differ even at soft_pca_alpha=1.

soft_pca_alpha is deliberately NOT named `alpha` -- that name is already used
throughout this codebase for the whitening pre-emphasis coefficient (mode F's
fixed [1,-alpha] filter), a completely different thing. Per-sample alpha is
usually tiny (apply-every-sample decay compounds fast); the natural way to
pick it is from a target off-subspace decay time constant tau (seconds) via
soft_pca_alpha_from_tau(fs, tau) = 1 - exp(-1/(fs*tau)).
"""

import numpy as np
from scipy.signal import lfilter

from .base import AFCAlgorithm
from .pemafc_pca import load_pca_model
from .registry import register_algorithm
from .whitening import ARWhitener, ARWhitenerConfig, PEMAFCMode


def soft_pca_alpha_from_tau(fs: float, tau: float) -> float:
    """Per-sample soft-PCA decay fraction giving an off-subspace-component
    decay time constant of `tau` seconds at sample rate `fs`."""
    return 1.0 - np.exp(-1.0 / (fs * tau))


class PEMAFCSoftPCA(AFCAlgorithm):
    def __init__(self, pca_model_path: str, mu: float, delta: float, mode: PEMAFCMode,
                 soft_pca_alpha: float, alpha: float = 0.9, ar_order: int = 12,
                 ar_block_len: int = 640, ar_reg: float = 1e-5,
                 n_components_active: int | None = None, name_override: str | None = None):
        self._afl, self.mean_ir, self.components = load_pca_model(pca_model_path)
        self.n_components_loaded = len(self.components)
        self.mu = mu
        self.delta = delta
        self.soft_pca_alpha = soft_pca_alpha
        self.whitener = ARWhitener(ARWhitenerConfig(
            mode=mode, ar_order=ar_order, alpha=alpha,
            ar_block_len=ar_block_len, ar_reg=ar_reg))
        self._mode = mode
        self._name_override = name_override
        self._hist_len = self._afl + max(ar_order, 1) + 8
        self._n_active_init = (self.n_components_loaded if n_components_active is None
                                else n_components_active)
        self.reset()

    def reset(self) -> None:
        self.whitener.reset()
        self._spk_hist = np.zeros(self._hist_len)
        self._mic_hist = np.zeros(self._hist_len)
        self.n_components_active = min(self._n_active_init, self.n_components_loaded)
        self.efbp = self.mean_ir.copy()

    def get_estimated_ir(self) -> np.ndarray:
        return self.efbp.copy()

    def push_reference_sample(self, true_spk_sample: float) -> None:
        self._spk_hist[-1] = true_spk_sample

    @property
    def name(self) -> str:
        base = self._name_override or f"SoftPCA-{self._mode.value}"
        return f"{base} (alpha={self.soft_pca_alpha:.2e}, K={self.n_components_active}/{self.n_components_loaded})"

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
            # see pemafc.py's process_block: a re-estimation boundary can fall
            # mid-call, so history must be carried forward after every
            # sub-chunk, not just once at the end of this call.
            self._spk_hist = np.concatenate([self._spk_hist, spk_chunk])[-self._hist_len:]
            self._mic_hist = np.concatenate([self._mic_hist, mic_chunk])[-self._hist_len:]
            pos += chunk_len

        return e_out

    def _process_chunk(self, mic_chunk: np.ndarray, spk_chunk: np.ndarray) -> np.ndarray:
        H = self._hist_len
        L = len(mic_chunk)
        extended_spk = np.concatenate([self._spk_hist, spk_chunk])
        extended_mic = np.concatenate([self._mic_hist, mic_chunk])
        b = np.concatenate([[1.0], self.whitener.ar_coeffs])
        filt_spk = lfilter(b, [1.0], extended_spk)
        filt_mic = lfilter(b, [1.0], extended_mic)

        afl = self._afl
        na = self.n_components_active
        active_components = self.components[:na]  # (na, afl)
        soft_alpha = self.soft_pca_alpha
        e_chunk = np.empty(L)
        for k in range(L):
            idx = H + k
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
            efbp_raw = self.efbp + mu_mult * u_f   # ordinary, unconstrained NLMS step

            if na > 0 and soft_alpha != 0.0:
                z = efbp_raw - self.mean_ir
                a = active_components @ z              # (na,)
                z_parallel = a @ active_components      # (afl,)
                z_perp = z - z_parallel
                self.efbp = efbp_raw - soft_alpha * z_perp
            else:
                self.efbp = efbp_raw

        self.whitener.push_errors(e_chunk)
        return e_chunk


@register_algorithm("soft_pca")
def _make_soft_pca(**kwargs) -> PEMAFCSoftPCA:
    kwargs.setdefault("mode", PEMAFCMode.F)
    return PEMAFCSoftPCA(**kwargs)
