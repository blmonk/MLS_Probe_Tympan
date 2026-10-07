"""
BlendPCA: a true combination of two INDEPENDENTLY adapting filters -- the
full-rank PEMAFCFull (mode F) and the PCA-constrained PEMAFCPCA (mode F) --
blended only at the output, as discussed in the conversation record (the
standard "combination of adaptive filters" technique: Arenas-Garcia et al.
style convex combination, blending at output only, each component filter
tracking the true system as if it alone were the canceller).

This is fundamentally different from PEMAFCSoftPCA (pemafc_soft_pca.py),
which maintains only ONE coefficient state and continuously shrinks its
off-subspace component -- there, "the PCA candidate" is just a geometric
projection of the single ongoing trajectory, never an independently-adapted
solution. Here there are genuinely TWO separate, independently-evolving
coefficient vectors (efbp_full, afl-dim; c_pca, K-dim), each updated using
its OWN prediction error, each blind to the other and to the blended output.

Real-device feasibility (see conversation record for the full discussion):
  - Memory: the two states are NOT two full afl-length vectors -- efbp_full
    is afl taps, but the PCA side only needs its own K-dim c_pca (tiny,
    e.g. 20 floats); the shared mean_ir/components basis is a fixed,
    read-only table loaded once regardless of how many filters reference it.
  - The regressor/mic sample history (self._spk_hist/_mic_hist) and the
    whitening filter (self.whitener) are SHARED, single copies -- both
    sub-filters read the same incoming audio and (since both use mode F,
    a fixed pre-emphasis) the same fixed whitening coefficient. Nothing
    about either sub-filter's input processing needs duplicating.
  - Compute: each sample does one O(afl) full-rank NLMS step plus one
    O(K*afl) PCA-projected step plus two extra dot products for the blended
    prediction -- roughly 2-3x a single algorithm's per-sample cost, which
    the existing Tympan hardware this project targets already has headroom
    for (it already runs the full afl=512 update in real time; the PCA
    side's K~12-20 dim update is considerably cheaper than that baseline).

Per-sample order (mirrors PEMAFCFull/PEMAFCPCA's shared whitening-then-NLMS
structure -- see pemafc.py's module docstring for why this ordering matters):
  1. fb_est_full = dot(u_raw, efbp_full); fb_est_pca = dot(u_raw, efbp_pca)
  2. fb_est_blend = blend_weight*fb_est_pca + (1-blend_weight)*fb_est_full
     e_out[k] = mic[k] - fb_est_blend          (the actual device output)
  3. whiten u_raw/mic (ONE shared whitening filter, both sub-filters read it)
  4. full filter's OWN independent NLMS step, using ITS OWN e_full_f
  5. PCA filter's OWN independent projected-NLMS step, using ITS OWN e_pca_f
  (steps 4 and 5 never reference each other's error or state -- this is
  exactly what "independent histories" means)

blend_weight=0.0 reduces to exactly PEMAFCFull alone (bit-exact, verified in
afc_sim/tests); blend_weight=1.0 reduces to exactly PEMAFCPCA alone.
get_estimated_ir() returns the BLENDED efbp (what's actually subtracted from
mic in a real device), which is what misalignment/MSG/ASG are measured
against -- not either sub-filter's own, individually "true" estimate.
"""

import numpy as np
from scipy.signal import lfilter

from .base import AFCAlgorithm
from .pemafc_pca import load_pca_model
from .registry import register_algorithm
from .whitening import ARWhitener, ARWhitenerConfig, PEMAFCMode


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)  # dynamic mode's blend_weight_init=0 or 1 would otherwise divide by zero
    return float(np.log(p / (1.0 - p)))


class BlendPCA(AFCAlgorithm):
    def __init__(self, pca_model_path: str, mu: float, delta: float,
                 blend_weight: float = 0.5, mode: PEMAFCMode = PEMAFCMode.F,
                 alpha: float = 0.9, ar_order: int = 12, ar_block_len: int = 640,
                 ar_reg: float = 1e-5, n_components_active: int | None = None,
                 name_override: str | None = None,
                 dynamic: bool = False, mu_lambda: float = 0.0,
                 blend_weight_init: float = 0.5, lambda_reg: float | None = None):
        """
        dynamic=False (default): blend_weight is a fixed constant, exactly as
        originally implemented -- see module docstring.

        dynamic=True: blend_weight is itself adapted every sample via the
        standard "combination of adaptive filters" technique (Arenas-Garcia
        et al.) -- gradient descent on the BLENDED output's own error,
        through a sigmoid so it stays smoothly in (0,1):

          w = sigmoid(a)                              (a: unconstrained logit)
          fb_est_blend = (1-w)*fb_est_full + w*fb_est_pca
          e_blend      = mic - fb_est_blend
          diff         = fb_est_pca - fb_est_full
          a += (mu_lambda / (diff**2 + lambda_reg)) * e_blend * w*(1-w) * diff

        This self-tunes toward whichever sub-filter is CURRENTLY reducing the
        blended error faster -- no hand-picked "what counts as large error"
        threshold needed, unlike a magnitude-heuristic switch (see
        conversation record for why that was rejected in favor of this).
        Operates on the RAW (unwhitened) blended error/predictions, matching
        the standard formulation -- whitening is specifically an NLMS
        update-conditioning technique for the two sub-filters, not something
        the combination-weight's own gradient needs.

        mu_lambda=0 degenerates to exactly the fixed-blend_weight case at
        whatever blend_weight_init was given (verified in afc_sim/tests) --
        a never moves, so w stays pinned at sigmoid(logit(blend_weight_init))
        == blend_weight_init.
        """
        self._afl, self.mean_ir, self.components = load_pca_model(pca_model_path)
        self.n_components_loaded = len(self.components)
        self.mu = mu
        self.delta = delta
        self.dynamic = dynamic
        self.mu_lambda = mu_lambda
        self.lambda_reg = lambda_reg if lambda_reg is not None else delta
        self._blend_weight_init = blend_weight_init if dynamic else blend_weight
        self.blend_weight = self._blend_weight_init
        # ONE shared whitener -- both sub-filters read the same fixed
        # pre-emphasis (mode F) rather than each maintaining its own AR state.
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
        self._spk_hist = np.zeros(self._hist_len)   # single shared history
        self._mic_hist = np.zeros(self._hist_len)
        self.blend_weight = self._blend_weight_init
        # _a (the logit) is only meaningful/used in dynamic mode -- the fixed
        # case leaves blend_weight exactly as set and never reads _a, so
        # don't compute logit(0) / logit(1) (divide-by-zero) for fixed
        # blend_weight=0 or 1, which are valid, common boundary-test values.
        self._a = _logit(self._blend_weight_init) if self.dynamic else None
        self.efbp_full = np.zeros(self._afl)         # full-rank filter's OWN state
        self.c_pca = np.zeros(self.n_components_loaded)   # PCA filter's OWN state
        self.n_components_active = min(self._n_active_init, self.n_components_loaded)
        self.efbp_pca = self.mean_ir.copy()

    def get_estimated_ir(self) -> np.ndarray:
        """The BLENDED estimate -- what a real device would actually subtract
        from mic, and what misalignment/MSG/ASG should be measured against."""
        w = self.blend_weight
        return (1.0 - w) * self.efbp_full + w * self.efbp_pca

    def push_reference_sample(self, true_spk_sample: float) -> None:
        self._spk_hist[-1] = true_spk_sample

    @property
    def name(self) -> str:
        base = self._name_override or f"BlendPCA-{self._mode.value}"
        if self.dynamic:
            return (f"{base} (dynamic, w0={self._blend_weight_init:.2f}, "
                    f"mu_lambda={self.mu_lambda:.1e}, K={self.n_components_active}/{self.n_components_loaded})")
        return (f"{base} (w={self.blend_weight:.2f}, "
                f"K={self.n_components_active}/{self.n_components_loaded})")

    def get_diagnostics(self) -> dict:
        return {"blend_weight": self.blend_weight} if self.dynamic else {}

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
        e_chunk = np.empty(L)
        # We re-use one error-signal accumulator for the whitener's AR
        # re-estimation (mode A only); feed it the BLENDED output error,
        # since that's the one real signal a real device would actually
        # have available to re-estimate coloration from.
        e_blend_chunk = np.empty(L)

        for k in range(L):
            idx = H + k
            lo = idx - afl
            hi = idx
            u_raw = extended_spk[lo:hi][::-1]

            w = self.blend_weight   # current value; dynamic mode updates it below for next sample
            fb_est_full = float(np.dot(u_raw, self.efbp_full))
            fb_est_pca = float(np.dot(u_raw, self.efbp_pca))
            fb_est_blend = (1.0 - w) * fb_est_full + w * fb_est_pca
            e_blend = mic_chunk[k] - fb_est_blend
            e_chunk[k] = e_blend
            e_blend_chunk[k] = e_blend

            u_f = filt_spk[lo:hi][::-1]
            y_f = filt_mic[idx]

            # Full-rank filter's OWN independent NLMS step -- uses ITS OWN
            # error (as if it alone were the canceller), never e_blend.
            fb_est_full_f = float(np.dot(u_f, self.efbp_full))
            e_full_f = y_f - fb_est_full_f
            norm_sq_full = float(np.dot(u_f, u_f)) + self.delta
            self.efbp_full += (self.mu / norm_sq_full) * e_full_f * u_f

            # PCA-constrained filter's OWN independent step -- same pattern.
            if na > 0:
                fb_est_pca_f = float(np.dot(u_f, self.efbp_pca))
                e_pca_f = y_f - fb_est_pca_f
                z = active_components @ u_f
                norm_sq_pca = float(np.dot(z, z)) + self.delta
                mu_mult = (self.mu / norm_sq_pca) * e_pca_f
                dc = mu_mult * z
                self.c_pca[:na] += dc
                self.efbp_pca += dc @ active_components

            if self.dynamic:
                # Gradient descent on the BLENDED (raw, unwhitened) error
                # through the sigmoid -- see __init__'s docstring. Uses
                # fb_est_full/fb_est_pca (already computed above, from the
                # SAME efbp_full/efbp_pca this sample used for prediction --
                # not the just-updated ones, which belong to the next sample).
                diff = fb_est_pca - fb_est_full
                norm_sq_lambda = diff * diff + self.lambda_reg
                grad = e_blend * w * (1.0 - w) * diff
                self._a += (self.mu_lambda / norm_sq_lambda) * grad
                self.blend_weight = _sigmoid(self._a)

        self.whitener.push_errors(e_blend_chunk)
        return e_chunk


@register_algorithm("blend_pca")
def _make_blend_pca(**kwargs) -> BlendPCA:
    kwargs.setdefault("mode", PEMAFCMode.F)
    return BlendPCA(**kwargs)


@register_algorithm("blend_pca_dynamic")
def _make_blend_pca_dynamic(**kwargs) -> BlendPCA:
    kwargs.setdefault("mode", PEMAFCMode.F)
    kwargs["dynamic"] = True
    return BlendPCA(**kwargs)
