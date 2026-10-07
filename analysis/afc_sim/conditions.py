"""
Loads impulse-response "conditions" from an mls_impulse.py --save_ir .npz
and splits them into disjoint train/held-out sets.

Two distinct meanings of "condition" this module supports without caring
which one it's given:
  - Individual per-period IRs (rows of `all_irs`) from ONE recording --
    today's only real data, a pipeline-validation stand-in (see
    afc_sim_PLAN.md Β§2) with within-session noise variation, not genuine
    cross-condition acoustic diversity.
  - Once available: one IR per physically distinct recording session
    (multiple .npz files, each contributing its own `mean_ir`) -- the
    scientifically meaningful case. `load_conditions_from_npz_files` handles
    this without needing any changes to the rest of afc_sim.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditionSet:
    fs: float
    mls_length: int
    ir_by_id: dict           # condition_id (str) -> raw IR (np.ndarray, length mls_length)
    quality_by_id: dict      # condition_id -> local SNR score (higher = cleaner)
    train_ids: list
    eval_ids: list


def _local_snr_scores(irs: np.ndarray, peak_guard: int = 40) -> np.ndarray:
    """
    Per-row local SNR: peak amplitude near the array's dominant (mean-IR)
    location vs. the median |amplitude| everywhere outside a guard band
    around it. Generic to any mls_length/peak location -- doesn't hardcode
    this recording's specific ~7ms/337-sample peak.
    """
    mean_ir = irs.mean(axis=0)
    peak = int(np.argmax(np.abs(mean_ir)))
    n = irs.shape[1]
    mask = np.ones(n, dtype=bool)
    lo, hi = max(0, peak - peak_guard), min(n, peak + peak_guard)
    mask[lo:hi] = False

    main_amp = np.max(np.abs(irs[:, max(0, peak - 10):min(n, peak + 10)]), axis=1)
    noise_lvl = np.median(np.abs(irs[:, mask]), axis=1) + 1e-15
    return main_amp / noise_lvl


def load_conditions_from_periods(npz_path: str, n_train: int = 40, n_eval: int = 6,
                                  seed: int = 0, min_quality_percentile: float = 50.0,
                                  n_average: int = 1) -> ConditionSet:
    """
    Build a ConditionSet from one recording's individual per-period IRs
    (all_irs). Keeps only (averaged) periods at/above `min_quality_percentile`
    local SNR, then deterministically (seeded) splits a disjoint train/eval
    subset from that quality-filtered pool -- so a low-quality period (its
    peak_idx landing on a noise spike) can't poison a held-out "ground truth".

    n_average > 1: each "condition" is the mean of `n_average` CONSECUTIVE,
    NON-OVERLAPPING raw periods (periods[0:n_average], periods[n_average:2*
    n_average], ...) rather than one raw period alone -- trades fewer total
    candidate conditions (len(all_irs) // n_average) for lower measurement
    noise per condition, same idea as this dataset's own repeated-MLS-period
    averaging. Non-overlapping by construction, so two chosen conditions can
    never share a raw period; quality scoring (_local_snr_scores) is computed
    on the AVERAGED array, not the raw one, since that's what's actually used
    as ground truth from here on -- an individually mediocre raw period can
    still contribute to a good averaged condition, and vice versa.
    """
    d = np.load(npz_path)
    fs = float(d["fs"])
    mls_length = int(d["mls_length"])
    raw_irs = np.asarray(d["all_irs"], dtype=np.float64)

    if n_average > 1:
        n_groups = len(raw_irs) // n_average
        if n_groups < n_train + n_eval:
            raise ValueError(
                f"Only {n_groups} groups of {n_average} consecutive periods available "
                f"from {len(raw_irs)} raw periods, need {n_train + n_eval} "
                f"(n_train={n_train} + n_eval={n_eval}). Lower n_average or n_train/n_eval.")
        irs = raw_irs[:n_groups * n_average].reshape(n_groups, n_average, -1).mean(axis=1)
    else:
        irs = raw_irs

    scores = _local_snr_scores(irs)
    thresh = np.percentile(scores, min_quality_percentile)
    good = np.where(scores >= thresh)[0]
    if len(good) < n_train + n_eval:
        raise ValueError(
            f"Only {len(good)} (averaged) periods pass the quality filter "
            f"(percentile={min_quality_percentile}), need {n_train + n_eval} "
            f"(n_train={n_train} + n_eval={n_eval}). Lower min_quality_percentile "
            f"or n_train/n_eval.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(good, size=n_train + n_eval, replace=False)
    train_idx = chosen[:n_train]
    eval_idx = chosen[n_train:]

    def _cid(i):
        if n_average > 1:
            lo, hi = i * n_average, i * n_average + n_average - 1
            return f"periodavg_{lo:04d}-{hi:04d}"
        return f"period_{i:04d}"

    ir_by_id = {}
    quality_by_id = {}
    for i in np.concatenate([train_idx, eval_idx]):
        cid = _cid(i)
        ir_by_id[cid] = irs[i]
        quality_by_id[cid] = float(scores[i])

    return ConditionSet(
        fs=fs, mls_length=mls_length, ir_by_id=ir_by_id, quality_by_id=quality_by_id,
        train_ids=[_cid(i) for i in train_idx],
        eval_ids=[_cid(i) for i in eval_idx],
    )


def load_conditions_from_npz_files(train_paths: list, eval_paths: list) -> ConditionSet:
    """
    The scientifically meaningful case: each path is its own physically
    distinct recording (.npz from mls_impulse.py --save_ir), contributing
    its `mean_ir` as one condition. train_paths / eval_paths must be
    disjoint files.
    """
    fs = mls_length = None
    ir_by_id, quality_by_id = {}, {}
    train_ids, eval_ids = [], []
    for group, ids_out in ((train_paths, train_ids), (eval_paths, eval_ids)):
        for path in group:
            d = np.load(path)
            this_fs, this_mls_length = float(d["fs"]), int(d["mls_length"])
            if fs is None:
                fs, mls_length = this_fs, this_mls_length
            elif this_fs != fs:
                raise ValueError(f"{path} has fs={this_fs}, expected {fs}")
            cid = path
            ir_by_id[cid] = np.asarray(d["mean_ir"], dtype=np.float64)
            quality_by_id[cid] = float("inf")  # a dedicated recording is trusted as-is
            ids_out.append(cid)
    return ConditionSet(fs=fs, mls_length=mls_length, ir_by_id=ir_by_id,
                         quality_by_id=quality_by_id, train_ids=train_ids, eval_ids=eval_ids)


def zero_lag_window(ir: np.ndarray, afl: int, lag: int = 1) -> np.ndarray:
    """
    raw_ir[lag:lag+afl], zero-padded if shorter -- the windowing convention
    this whole simulator uses (NOT train_pca_feedback_model.py's peak-anchored
    window_around_peak -- see afc_sim_PLAN.md Β§4).

    `lag` must match whatever causal-lag convention produced the regressor
    being compared against -- getting it wrong reintroduces the same class of
    bug as the earlier peak-anchored `pre_peak` windowing mismatch, just by a
    sample or two instead of ~330:

    - lag=1 (default): open-loop ground truth (also used for PCA training,
      which is always fit against open-loop-style targets). FeedbackPath's
      convolution includes ir[0] as a same-instant tap (ir[k] = gain from
      spk[i-k]) that the algorithm's own 1-sample-lag regressor
      (offset_u[m] = spk[i-1-m]) can never reach, so efbp[m] targets
      ir[m+1], not ir[m].
    - lag=0: closed-loop ground truth. closed_loop.py's own per-sample loop
      already imposes the minimum 1-sample lag directly when consuming this
      same array (mic[i] = desired[i] + sum_j true_ir[j]*played[i-1-j] --
      true_ir[j] IS the coefficient multiplying played[i-1-j]), so
      efbp[m] must equal true_ir[m] with NO further shift: both sums range
      over the identical regressor terms played[i-1-m]. Applying the
      open-loop lag=1 shift on top of this double-shifts the target by one
      sample (confirmed by direct hand-trace of run_closed_loop_epoch with a
      synthetic true_ir and NullAFC).
    """
    out = np.zeros(afl)
    available = ir[lag:lag + afl]
    out[:len(available)] = available
    return out
