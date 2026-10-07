"""Persist simulation results: a tidy CSV (one row per metrics sample, so
comparing across many future runs never requires re-parsing plots), a run
manifest (config + timestamp), and comparison plots."""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np


def save_results_csv(results, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "epoch_index", "condition_id", "t_s", "misalignment_db", "erle_db",
                    "msg_with_afc_db", "msg_without_afc_db", "asg_db"])
        for r in results:
            # msg_without_afc_db is constant per trial but repeated per row --
            # consistent with this file's existing denormalized style (see
            # save_manifest's own docstring on why: hand-picking what's
            # redundant is exactly how source_npz_path went missing once).
            for t, mis, erle, msg_with, asg in zip(
                    r.t, r.misalignment_db, r.erle_db, r.msg_with_afc_db, r.asg_db):
                w.writerow([r.algorithm_name, r.epoch_index, r.condition_id,
                            f"{t:.6f}", f"{mis:.4f}", f"{erle:.4f}",
                            f"{msg_with:.4f}", f"{r.msg_without_afc_db:.4f}", f"{asg:.4f}"])


def save_manifest(results, config, conditions, out_path: str, source_npz_path: str = None,
                   cli_args: dict = None, algorithm_names: list = None,
                   pca_fit_info: dict = None) -> None:
    """
    cli_args: the full argparse Namespace as a dict (vars(args)), saved
    as-is rather than hand-picking individual fields -- hand-picking is
    exactly how source_npz_path went missing the first time; saving
    everything the CLI was actually invoked with guarantees this doesn't
    happen again for the next parameter someone adds.
    """
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_npz_path": source_npz_path,
        "cli_args": cli_args,
        "algorithm_names": algorithm_names,
        "pca_fit_info": pca_fit_info,
        "config": {
            "fs": config.fs,
            "block_size": config.block_size,
            "forward_signal_spec": asdict(config.forward_signal_spec),
            "epochs": [asdict(e) for e in config.epochs],
            "metrics_interval_s": config.metrics_interval_s,
            "reset_between_epochs": config.reset_between_epochs,
            "misalignment_threshold_db": config.misalignment_threshold_db,
            "loop_mode": config.loop_mode,
            "gain_db": config.gain_db,
            "saturation_limit": config.saturation_limit,
        },
        "train_condition_ids": conditions.train_ids,
        "eval_condition_ids": conditions.eval_ids,
        "convergence_time_s": {
            f"{r.algorithm_name}/epoch{r.epoch_index}_{r.condition_id}": r.convergence_time_s
            for r in results
        },
        "saturated_fraction": {
            f"{r.algorithm_name}/epoch{r.epoch_index}_{r.condition_id}": r.saturated_fraction
            for r in results
        } if config.loop_mode == "closed" else None,
        "msg_without_afc_db": {
            # Keyed exactly like convergence_time_s/saturated_fraction above, not by
            # condition_id alone: MSG_without_AFC depends on (true_ir_afl, afl), and
            # afl is set per-algorithm -- today's CLI happens to apply one global
            # --afl to every variant, but that's a CLI wiring choice, not a
            # framework invariant, so keying by the full tuple is correct in
            # general and costs nothing when afl matches across algorithms.
            f"{r.algorithm_name}/epoch{r.epoch_index}_{r.condition_id}": r.msg_without_afc_db
            for r in results
        },
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _algo_colors(results):
    algo_names = sorted(set(r.algorithm_name for r in results))
    import matplotlib.pyplot as plt
    return algo_names, dict(zip(algo_names, plt.cm.tab10(np.linspace(0, 0.9, len(algo_names)))))


def _plot_single_series(results, config, out_path: str, get_series, ylabel: str, title: str) -> None:
    """One metric, one file, one line per algorithm -- shared boilerplate for
    the simple (no per-epoch reference line) plots: misalignment (dB and
    linear), ERLE, ASG."""
    import matplotlib.pyplot as plt

    algo_names, colors = _algo_colors(results)
    switch_times = np.cumsum([0.0] + [e.duration_s for e in config.epochs[:-1]])

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in algo_names:
        trials = sorted((r for r in results if r.algorithm_name == name), key=lambda r: r.epoch_index)
        t_all = np.concatenate([r.t for r in trials])
        val_all = np.concatenate([get_series(r) for r in trials])
        ax.plot(t_all, val_all, label=name, color=colors[name], lw=1.2)

    for st in switch_times:
        ax.axvline(st, color="k", lw=0.8, ls="--", alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_misalignment_db(results, config, out_path: str) -> None:
    _plot_single_series(results, config, out_path, lambda r: r.misalignment_db,
                         "Misalignment (dB)",
                         "Feedback-path estimate misalignment vs. time (dashed = condition switch)")


def plot_misalignment_linear(results, config, out_path: str) -> None:
    # misalignment_db = 20*log10(ratio)
    _plot_single_series(results, config, out_path, lambda r: 10.0 ** (r.misalignment_db / 20.0),
                         "Misalignment (ratio)",
                         "Same misalignment, linear scale (||true - est|| / ||true||)")


def plot_erle(results, config, out_path: str) -> None:
    _plot_single_series(results, config, out_path, lambda r: r.erle_db,
                         "ERLE (dB)", "Echo/feedback return loss enhancement vs. time")


def plot_asg(results, config, out_path: str) -> None:
    _plot_single_series(results, config, out_path, lambda r: r.asg_db,
                         "ASG (dB)", "Added Stable Gain vs. time (MSG improvement over no-AFC baseline)")


def plot_msg(results, config, out_path: str) -> None:
    """Separate from plot_asg: MSG_without_AFC is a per-epoch constant
    needing a different idiom (horizontal dashed reference segments) than
    the smooth time series the other plots use."""
    import matplotlib.pyplot as plt

    algo_names, colors = _algo_colors(results)
    switch_times = list(np.cumsum([0.0] + [e.duration_s for e in config.epochs[:-1]])) + \
        [sum(e.duration_s for e in config.epochs)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in algo_names:
        trials = sorted((r for r in results if r.algorithm_name == name), key=lambda r: r.epoch_index)
        t_all = np.concatenate([r.t for r in trials])
        msg_all = np.concatenate([r.msg_with_afc_db for r in trials])
        ax.plot(t_all, msg_all, label=name, color=colors[name], lw=1.2)

    # "no AFC" is a property of the (condition, afl) pair, not of any
    # algorithm -- draw it once per epoch in a neutral color, not once per
    # algorithm in that algorithm's own color. Different algorithms normally
    # share one global --afl and therefore agree exactly on this value per
    # epoch; if a future run mixes afl across algorithms they could
    # legitimately disagree, so group by the actual value rather than assume.
    by_epoch = sorted(set(r.epoch_index for r in results))
    for i, epoch_idx in enumerate(by_epoch):
        t0, t1 = switch_times[i], switch_times[i + 1]
        epoch_trials = [r for r in results if r.epoch_index == epoch_idx]
        distinct_vals = sorted(set(round(r.msg_without_afc_db, 6) for r in epoch_trials))
        for val in distinct_vals:
            label = "no AFC" if (i == 0 and len(distinct_vals) == 1) else f"no AFC ({val:.1f}dB)" if i == 0 else None
            ax.plot([t0, t1], [val] * 2, color="black", lw=1.2, ls=":", alpha=0.7, label=label)

    for st in switch_times[:-1]:
        ax.axvline(st, color="k", lw=0.8, ls="--", alpha=0.5)

    ax.set_ylabel("MSG (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Maximum Stable Gain vs. time (solid = with AFC, dotted = no AFC "
                  "reference, dashed = condition switch)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_diagnostics(results, config, out_dir: str) -> list:
    """Plots any algorithm-specific extra diagnostics (AFCAlgorithm.get_diagnostics(),
    e.g. BlendPCA's dynamically-adapted blend_weight) -- one FILE per distinct
    diagnostic key seen (diagnostics_<key>.png), one line per algorithm that
    actually reports that key (most won't report anything). Returns the list
    of paths written -- empty if no result has any diagnostics, so a run
    with no diagnostic-reporting algorithms produces no file at all."""
    import matplotlib.pyplot as plt

    keys = sorted({k for r in results for k in r.diagnostics})
    if not keys:
        return []

    algo_names, colors = _algo_colors(results)
    switch_times = np.cumsum([0.0] + [e.duration_s for e in config.epochs[:-1]])

    written = []
    for key in keys:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in algo_names:
            trials = sorted((r for r in results if r.algorithm_name == name and key in r.diagnostics),
                             key=lambda r: r.epoch_index)
            if not trials:
                continue
            t_all = np.concatenate([r.t for r in trials])
            val_all = np.concatenate([r.diagnostics[key] for r in trials])
            ax.plot(t_all, val_all, label=name, color=colors[name], lw=1.2)
        for st in switch_times:
            ax.axvline(st, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(key)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{key} vs. time (dashed = condition switch)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(out_dir, f"diagnostics_{key}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        written.append(out_path)
    return written


def save_all(results, config, conditions, out_dir: str, source_npz_path: str = None,
             cli_args: dict = None, algorithm_names: list = None,
             pca_fit_info: dict = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_results_csv(results, os.path.join(out_dir, "results.csv"))
    save_manifest(results, config, conditions, os.path.join(out_dir, "manifest.json"),
                  source_npz_path=source_npz_path, cli_args=cli_args,
                  algorithm_names=algorithm_names, pca_fit_info=pca_fit_info)
    plot_misalignment_db(results, config, os.path.join(out_dir, "misalignment_db.png"))
    plot_misalignment_linear(results, config, os.path.join(out_dir, "misalignment_linear.png"))
    plot_erle(results, config, os.path.join(out_dir, "erle.png"))
    plot_msg(results, config, os.path.join(out_dir, "msg.png"))
    plot_asg(results, config, os.path.join(out_dir, "asg.png"))
    plot_diagnostics(results, config, out_dir)
