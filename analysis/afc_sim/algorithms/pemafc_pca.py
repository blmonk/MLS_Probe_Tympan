"""
PEMAFCPCA: PCA-constrained port of AudioFeedbackCancelPEMAFC_ConstrainedPCA_F32.

Reparametrization: efbp = mean_ir + sum_k c[k]*component[k], where mean_ir
and components (unit-norm rows) are FIXED (loaded from a pca_model.csv);
only the K-dim c[] adapts. The whitening front end (steps 1-4 below) is
identical to PEMAFCFull -- see algorithms/pemafc.py's module docstring for
the full step ordering. Only the update step differs:

  z[k]    = dot(u_f, component[k])           for k in 0..n_active-1
  norm_sq = sum(z[k]**2) + delta
  mu_mult = (mu/norm_sq) * e_f
  c[k]   += mu_mult * z[k]
  efbp   += mu_mult * z[k] * component[k]     (folded incrementally, not
                                                recomputed from mean+c@comp
                                                every sample)

n_components_active in [0, n_components_loaded] is independently settable
from the loaded model's own K, so a fitted model can be swept from "fully
constrained to the PCA subspace" (n_active = K) down to "pinned at mean_ir,
no adaptation at all" (n_active = 0) without refitting.
"""

import numpy as np
from scipy.signal import lfilter

from .base import AFCAlgorithm
from .registry import register_algorithm
from .whitening import ARWhitener, ARWhitenerConfig, PEMAFCMode


def load_pca_model(path: str):
    """Parse a pca_model.csv (afl,K header; mean_ir row; K component rows).
    Returns (afl, mean_ir, components) with components shape (K, afl)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    afl, K = (int(x) for x in lines[0].split(","))
    mean_ir = np.array([float(x) for x in lines[1].split(",")])
    components = np.array([[float(x) for x in lines[2 + k].split(",")] for k in range(K)])
    assert mean_ir.shape == (afl,)
    assert components.shape == (K, afl)
    return afl, mean_ir, components


class PEMAFCPCA(AFCAlgorithm):
    def __init__(self, pca_model_path: str, mu: float, delta: float, mode: PEMAFCMode,
                 alpha: float = 0.9, ar_order: int = 12, ar_block_len: int = 640,
                 ar_reg: float = 1e-5, n_components_active: int | None = None,
                 name_override: str | None = None):
        self._afl, self.mean_ir, self.components = load_pca_model(pca_model_path)
        self.n_components_loaded = len(self.components)
        self.mu = mu
        self.delta = delta
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
        self.c = np.zeros(self.n_components_loaded)
        self.n_components_active = min(self._n_active_init, self.n_components_loaded)
        self._rebuild_efbp()

    def _rebuild_efbp(self) -> None:
        self.efbp = self.mean_ir + self.c @ self.components

    def set_num_components_active(self, k: int) -> int:
        k = max(0, min(k, self.n_components_loaded))
        self.n_components_active = k
        self.c[k:] = 0.0
        self._rebuild_efbp()
        return k

    def get_estimated_ir(self) -> np.ndarray:
        return self.efbp.copy()

    def push_reference_sample(self, true_spk_sample: float) -> None:
        self._spk_hist[-1] = true_spk_sample

    @property
    def name(self) -> str:
        base = self._name_override or f"PEMAFC-PCA-{self._mode.value}"
        return f"{base} (K={self.n_components_active}/{self.n_components_loaded})"

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

        afl, na = self._afl, self.n_components_active
        active_components = self.components[:na]  # (na, afl)
        e_chunk = np.empty(L)
        for k in range(L):
            idx = H + k
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

            if na > 0:
                z = active_components @ u_f                      # (na,)
                norm_sq = float(np.dot(z, z)) + self.delta
                mu_mult = (self.mu / norm_sq) * e_f
                dc = mu_mult * z
                self.c[:na] += dc
                self.efbp += dc @ active_components               # sum_k dc[k]*component[k]

        self.whitener.push_errors(e_chunk)
        return e_chunk


@register_algorithm("pemafc_pca_a")
def _make_pemafc_pca_a(**kwargs) -> PEMAFCPCA:
    kwargs.setdefault("mode", PEMAFCMode.A)
    return PEMAFCPCA(**kwargs)


@register_algorithm("pemafc_pca_f")
def _make_pemafc_pca_f(**kwargs) -> PEMAFCPCA:
    kwargs.setdefault("mode", PEMAFCMode.F)
    return PEMAFCPCA(**kwargs)


@register_algorithm("nlms_pca")
def _make_nlms_pca(**kwargs) -> PEMAFCPCA:
    # Plain (unwhitened) NLMS, PCA-constrained -- see pemafc.py's "nlms" for
    # why alpha=0/mode=F is an exact no-whitening identity.
    kwargs["mode"] = PEMAFCMode.F
    kwargs["alpha"] = 0.0
    kwargs["name_override"] = "NLMS-PCA"
    return PEMAFCPCA(**kwargs)
