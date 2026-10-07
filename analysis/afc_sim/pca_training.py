"""
Thin wrapper around train_pca_feedback_model's fit_pca()/write_pca_csv(),
using THIS simulator's zero-lag windowing convention (conditions.zero_lag_window)
instead of that script's peak-anchored window_around_peak -- see
afc_sim_PLAN.md Β§4 for why the two must not be mixed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_pca_feedback_model import fit_pca, write_pca_csv  # noqa: E402

from .conditions import ConditionSet, zero_lag_window  # noqa: E402


def train_pca_model(conditions: ConditionSet, afl: int, out_path: str,
                     n_components: int | None = None, var_explained: float = 0.95,
                     device_max_components: int = 16):
    """Fits a PCA model from conditions.train_ids's IRs (zero-lag windowed to
    afl) and writes it to out_path. Returns (mean_ir, components, ratio,
    cumulative, K) for callers that want the sanity-check plot in
    afc_sim_PLAN.md's Verification section."""
    if len(conditions.train_ids) < 2:
        raise ValueError(
            f"Need >=2 training conditions for PCA, got {len(conditions.train_ids)}")
    windows = [zero_lag_window(conditions.ir_by_id[cid], afl) for cid in conditions.train_ids]
    mean_ir, components, ratio, cumulative, K = fit_pca(
        windows, n_components=n_components, var_explained=var_explained,
        device_max_components=device_max_components)
    write_pca_csv(out_path, afl, mean_ir, components)
    return mean_ir, components, ratio, cumulative, K
