#!/usr/bin/env python3
"""
train_pca_feedback_model.py — Fit a PCA feedback-path model from multiple MLS
impulse-response measurements, for use by AudioFeedbackCancelPEMAFC_ConstrainedPCA_F32
(Tympan_Library) via its 'load <filename>' / loadPCAModelFromSD() SD card model.

Input: one or more .npz files produced by mls_impulse.py's --save_ir option,
each holding one measurement (fs, mls_length, mean_ir, all_irs, peak_idx).
Each .npz should come from a physically distinct condition (different earpiece
insertion depth, vent size, subject/coupler, environment, etc.) — the whole
point of PCA here is to characterize *plausible variation* in the feedback
path, so measurements that are just repeats of the exact same condition add
little. Aim for at least ~2-3x as many measurements as the number of PCA
components you want to keep.

Algorithm:
  1. Load every .npz, check fs is consistent across all of them.
  2. For each measurement's mean_ir, window a length-`afl` segment starting
     `--pre_peak` samples before that measurement's own impulse peak. Anchoring
     each window on its own peak (rather than an absolute sample index) aligns
     the measurements to a common delay reference before PCA, so PCA captures
     shape variation rather than being dominated by differences in absolute
     propagation delay between recordings.
  3. Stack the windows into a data matrix X (n_measurements x afl), compute
     mean_ir = X.mean(axis=0), and run PCA via SVD on the centered data.
  4. Keep the top K components, chosen either by --n_components or by the
     smallest K that reaches --var_explained cumulative explained variance
     (K is also clipped to --device_max_components, which must match
     MAX_PCA_COMPONENTS in AudioFeedbackCancelPEMAFC_ConstrainedPCA_F32.h).
  5. Write mean_ir + components to a CSV model file:
         afl,K
         <mean_ir[0]>,...,<mean_ir[afl-1]>
         <component0[0]>,...,<component0[afl-1]>
         ...
     Copy this file onto the Tympan's SD card and load it with the 'load'
     serial command (or automatically at startup — see the .ino).

Usage:
  python train_pca_feedback_model.py meas1.npz meas2.npz meas3.npz --afl 256
  python train_pca_feedback_model.py measurements/*.npz --afl 256 --n_components 6
  python train_pca_feedback_model.py measurements/*.npz --afl 256 --var_explained 0.99 --out pca_model.csv
"""

import argparse
import glob
import sys
import numpy as np


def load_measurement(path):
    d = np.load(path)
    for key in ("fs", "mls_length", "mean_ir"):
        if key not in d:
            sys.exit(f"ERROR: {path} is missing '{key}' — was it saved with "
                      f"mls_impulse.py --save_ir?")
    return float(d["fs"]), int(d["mls_length"]), np.asarray(d["mean_ir"], dtype=np.float64)


def window_around_peak(ir, afl, pre_peak):
    """
    Extract a length-afl window from `ir`, starting `pre_peak` samples before
    ir's own peak (by |amplitude|). Zero-pads if the window runs off either
    end of `ir`.
    """
    peak = int(np.argmax(np.abs(ir)))
    start = peak - pre_peak
    out = np.zeros(afl, dtype=np.float64)
    src_start = max(0, start)
    src_end   = min(len(ir), start + afl)
    dst_start = src_start - start
    dst_end   = dst_start + (src_end - src_start)
    if src_end > src_start:
        out[dst_start:dst_end] = ir[src_start:src_end]
    return out, peak


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz_files", nargs="+",
                        help="One or more .npz measurement files (glob patterns are "
                             "expanded automatically, e.g. 'measurements/*.npz')")
    parser.add_argument("--afl", type=int, required=True,
                        help="Adaptive filter length (taps) the constrained AFC will use "
                             "on the device. No default -- this fixes the model's afl.")
    parser.add_argument("--pre_peak", type=int, default=8,
                        help="Samples to keep before each measurement's impulse peak when "
                             "windowing to --afl (default: 8)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n_components", type=int, default=None,
                        help="Fixed number of PCA components to keep (overrides --var_explained)")
    group.add_argument("--var_explained", type=float, default=0.95,
                        help="Keep the smallest number of components reaching this cumulative "
                             "explained variance fraction (default: 0.95)")
    parser.add_argument("--device_max_components", type=int, default=16,
                        help="Clip K to this value -- must match MAX_PCA_COMPONENTS in "
                             "AudioFeedbackCancelPEMAFC_ConstrainedPCA_F32.h (default: 16)")
    parser.add_argument("--out", type=str, default="pca_model.csv",
                        help="Output CSV model file (default: pca_model.csv). Copy this to "
                             "the Tympan's SD card.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip the explained-variance / component plots.")
    args = parser.parse_args()

    # Expand glob patterns (works even if the shell already expanded some of them)
    paths = []
    for pattern in args.npz_files:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    if not paths:
        sys.exit("ERROR: no input files matched.")

    # ── Load and window every measurement ────────────────────────────────────
    # Measurements may use different MLS lengths (mls_length varies per
    # recording) -- that's fine here, since each mean_ir is independently
    # windowed to a common afl-length frame around its own peak. A too-short
    # recording is skipped (with a warning) rather than aborting the whole
    # batch, so one bad/short file doesn't discard everything else.
    fs_ref = None
    windows = []
    peaks_ms = []
    mls_lengths = []
    skipped = []
    for path in paths:
        fs, mls_length, mean_ir = load_measurement(path)
        if fs_ref is None:
            fs_ref = fs
        elif fs != fs_ref:
            sys.exit(f"ERROR: {path} has fs={fs} Hz, but earlier files used {fs_ref} Hz. "
                      f"All measurements must share the same sample rate.")
        if len(mean_ir) < args.afl:
            print(f"WARNING: skipping {path} -- its mean_ir has only {len(mean_ir)} samples "
                  f"(mls_length={mls_length}), shorter than --afl {args.afl}.")
            skipped.append(path)
            continue
        win, peak = window_around_peak(mean_ir, args.afl, args.pre_peak)
        windows.append(win)
        peaks_ms.append(1000.0 * peak / fs)
        mls_lengths.append(mls_length)
        print(f"Loaded : {path}  (mls_length={mls_length}, peak @ {1000.0*peak/fs:.3f} ms)")

    n = len(windows)
    if n < 2:
        sys.exit(f"ERROR: need at least 2 usable measurements to run PCA "
                 f"(got {n}{f', {len(skipped)} skipped as too short' if skipped else ''}).")

    if len(set(mls_lengths)) > 1:
        print(f"\nNote: measurements used {sorted(set(mls_lengths))} different MLS lengths. "
              f"That's fine for PCA -- each mean_ir is windowed to a common afl-length frame "
              f"around its own peak regardless of the source period length. But a *shorter* "
              f"MLS period gives circular deconvolution less room before wraparound: if the "
              f"true feedback path decays more slowly than the shortest period used, that "
              f"measurement's 'impulse response' may actually be aliased. Prefer a consistent "
              f"(and sufficiently long) MLS length across the training set when possible.")

    X = np.stack(windows, axis=0)   # (n, afl)

    print(f"\n{n} measurements, fs={fs_ref:.0f} Hz, afl={args.afl}, "
          f"pre_peak={args.pre_peak}")
    print(f"Peak delays (ms): min={min(peaks_ms):.3f}  max={max(peaks_ms):.3f}  "
          f"spread={max(peaks_ms)-min(peaks_ms):.3f}")
    if max(peaks_ms) - min(peaks_ms) > 1000.0 * args.pre_peak / fs_ref:
        print("WARNING: peak-delay spread exceeds --pre_peak worth of time; some "
              "measurements' true onsets may fall outside the window. Consider "
              "increasing --pre_peak.")

    # ── PCA via SVD ───────────────────────────────────────────────────────────
    mean_ir_model = X.mean(axis=0)
    Xc = X - mean_ir_model
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    n_avail = Vt.shape[0]
    explained_var = (S ** 2) / max(n - 1, 1)
    total_var = explained_var.sum()
    ratio = explained_var / total_var if total_var > 0 else explained_var
    cumulative = np.cumsum(ratio)

    if args.n_components is not None:
        K = min(args.n_components, n_avail)
    else:
        K = int(np.searchsorted(cumulative, args.var_explained) + 1)
        K = min(K, n_avail)
    if K > args.device_max_components:
        print(f"WARNING: requested/needed K={K} exceeds --device_max_components "
              f"{args.device_max_components}; clipping. Explained variance will be lower "
              f"than requested. Either collect more distinct measurements to reduce K's "
              f"requirement, or raise MAX_PCA_COMPONENTS on the device and re-run with a "
              f"higher --device_max_components.")
        K = args.device_max_components
    K = max(K, 1)

    components = Vt[:K]  # (K, afl), already unit-norm rows from SVD

    print("\nExplained variance by component:")
    for k in range(n_avail):
        marker = " <-- kept" if k < K else ""
        print(f"  PC{k:2d}: {ratio[k]*100:6.2f}%   cumulative {cumulative[k]*100:6.2f}%{marker}")
    print(f"\nKeeping K={K} components "
          f"(cumulative explained variance = {cumulative[K-1]*100:.2f}%)")

    # ── Write model file ──────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        f.write(f"{args.afl},{K}\n")
        f.write(",".join(f"{v:.8e}" for v in mean_ir_model) + "\n")
        for k in range(K):
            f.write(",".join(f"{v:.8e}" for v in components[k]) + "\n")

    print(f"\nWrote  : {args.out}")
    print(f"         Copy this file to the Tympan's SD card and load it with "
          f"'load {args.out}' (or set it as the default in the .ino).")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    t_ms = 1000.0 * np.arange(args.afl) / fs_ref

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(np.arange(n_avail), ratio * 100, color="steelblue", label="per-component")
    ax1.plot(np.arange(n_avail), cumulative * 100, color="tomato", marker="o",
             lw=1.2, label="cumulative")
    ax1.axhline(args.var_explained * 100, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.axvline(K - 0.5, color="k", lw=0.8, ls=":", alpha=0.6)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(f"PCA explained variance ({n} measurements)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(t_ms, mean_ir_model, color="black", lw=1.4, label="mean_ir")
    colors = plt.cm.tab10(np.linspace(0, 0.9, K))
    for k in range(K):
        ax2.plot(t_ms, components[k], lw=0.9, alpha=0.8, color=colors[k],
                 label=f"PC{k} ({ratio[k]*100:.1f}%)")
    ax2.axhline(0, color="k", lw=0.4)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Mean impulse response and PCA components")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
