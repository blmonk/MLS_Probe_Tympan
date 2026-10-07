#!/usr/bin/env python3
"""
run_experiment.py -- CLI entrypoint: trains a PCA model from a held-out
split of one recording's periods, builds the requested algorithm variants,
runs the open-loop switching simulation, and saves results.

Usage:
  python -m afc_sim.run_experiment ../data/self/AUDIO001_ir_hp100.npz --out-dir results/run1
  python -m afc_sim.run_experiment ../data/self/AUDIO001_ir_hp100.npz \\
      --variants pemafc_f,pemafc_a,pemafc_pca_a --epoch-duration 10 --n-eval 4

Caveat carried over from afc_sim_PLAN.md: with only one physical recording
session behind the "conditions" (see conditions.py), this validates that the
framework works end-to-end, not that PCA-constraining helps in reality --
that claim needs real distinct-condition recordings.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afc_sim import conditions as conditions_mod
from afc_sim import pca_training, results_io
from afc_sim.algorithms import create_algorithm
from afc_sim.feedback_path import FeedbackPath
from afc_sim.signals import SignalSpec
from afc_sim.simulate import EpochSpec, SimulationConfig, run_simulation
from afc_sim.algorithms.whitening import PEMAFCMode
from afc_sim.algorithms.pemafc_soft_pca import soft_pca_alpha_from_tau


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_path", help="Recording .npz from mls_impulse.py --save_ir")
    p.add_argument("--afl", type=int, default=512)
    p.add_argument("--block-size", type=int, default=640,
                    help="Open-loop batching granularity passed to algo.process_block() "
                         "per call. Must stay <= metrics_interval_s*fs (2400 samples at "
                         "the defaults) or misalignment/ERLE/MSG snapshots get "
                         "under-sampled (see simulate.py's _MetricsAccumulator.feed()). "
                         "640 sits at the speed plateau measured empirically (bigger "
                         "buys nothing further, since process_block's own internal "
                         "re-chunking around AR re-estimation boundaries already governs "
                         "the real granularity beyond this point); the old default of 32 "
                         "was ~20%% slower for no benefit.")
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--n-average", type=int, default=1,
                    help="Average this many consecutive, non-overlapping raw MLS periods "
                         "together to form each condition's ground-truth IR (default 1: use "
                         "each raw period alone). Reduces measurement noise per condition at "
                         "the cost of fewer total candidate conditions.")
    p.add_argument("--min-quality-percentile", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epoch-duration", type=float, default=20.0)
    p.add_argument("--signal", choices=["white", "pink", "colored", "wav", "ar_fit"], default="white")
    p.add_argument("--wav-path", type=str, default=None)
    p.add_argument("--signal-amplitude", type=float, default=0.1)
    p.add_argument("--signal-color-pole", type=float, default=0.9,
                    help="--signal colored only: AR(1) pole. 0.9 (default) matches PEMAFCFull's "
                         "own default --alpha, i.e. exactly the coloration mode F's fixed "
                         "pre-emphasis is designed to invert -- white noise structurally "
                         "disadvantages mode F (it adapts to nothing), this does the opposite.")
    p.add_argument("--ar-fit-start-s", type=float, default=0.0,
                    help="--signal ar_fit only: where in --wav-path to fit the AR envelope from.")
    p.add_argument("--ar-fit-duration-s", type=float, default=5.0,
                    help="--signal ar_fit only: length of that fit segment.")
    p.add_argument("--ar-fit-order", type=int, default=20,
                    help="--signal ar_fit only: LPC order for the fitted envelope. Generates "
                         "STATIONARY noise with the same long-term spectral shape as that real "
                         "segment (e.g. real speech) -- same coloration a whitening algorithm has "
                         "to deal with, but without speech's pauses/dynamic range swings that "
                         "otherwise make convergence curves noisy and hard to read.")
    p.add_argument("--variants", type=str, default="pemafc_f,pemafc_pca_f,nlms,nlms_pca")
    p.add_argument("--n-components-active", type=int, default=None,
                    help="Override the PCA variant's active component count (default: all loaded)")
    p.add_argument("--soft-pca-taus", type=str, default="0.05,0.1,0.3,1.0",
                    help="Comma-separated off-subspace decay time constants (seconds) for the "
                         "'soft_pca' variant -- each produces its own algorithm instance/plot "
                         "line, e.g. 'SoftPCA (alpha=2.08e-04, K=.../...)' for tau=0.1s at "
                         "fs=48kHz. Converted to a per-sample soft_pca_alpha via "
                         "soft_pca_alpha_from_tau(fs, tau) = 1 - exp(-1/(fs*tau)) -- see "
                         "pemafc_soft_pca.py. Ignored if --soft-pca-alphas is given. Only used "
                         "if 'soft_pca' is in --variants.")
    p.add_argument("--soft-pca-alphas", type=str, default=None,
                    help="Comma-separated per-sample soft_pca_alpha values directly (e.g. "
                         "'1e-5,1e-4,1e-3,1e-2,1e-1'), overriding --soft-pca-taus -- use this "
                         "when you want specific alpha magnitudes rather than ones derived from "
                         "a decay time constant. Only used if 'soft_pca' is in --variants.")
    p.add_argument("--blend-weights", type=str, default="0.5",
                    help="Comma-separated blend_weight values for the 'blend_pca' variant -- "
                         "each produces its own instance/plot line combining INDEPENDENTLY "
                         "adapting PEMAFC-F and PEMAFC-PCA-F (see blend_pca.py), weight w giving "
                         "(1-w)*PEMAFC-F + w*PEMAFC-PCA-F. Only used if 'blend_pca' is in "
                         "--variants.")
    p.add_argument("--blend-dynamic-mu-lambda", type=float, default=0.01,
                    help="'blend_pca_dynamic' only: step size for the gradient-adapted mixing "
                         "weight (see BlendPCA's dynamic mode docstring). 0 degenerates to a "
                         "fixed weight pinned at --blend-dynamic-weight-init.")
    p.add_argument("--blend-dynamic-weight-init", type=float, default=0.5,
                    help="'blend_pca_dynamic' only: starting blend_weight before adaptation.")
    p.add_argument("--blend-dynamic-lambda-reg", type=float, default=None,
                    help="'blend_pca_dynamic' only: regularizer for the mixing-weight gradient's "
                         "normalization (diff**2 + lambda_reg). Defaults to --delta, which is "
                         "tuned for ||u_f||^2-scale normalization -- a totally different scale "
                         "from diff**2 (the difference between two sub-filters' predictions). "
                         "For this project's real measured IRs (raw amplitude ~1e-6 to 1e-7), "
                         "diff**2 ~ 1e-12, so the --delta default (1e-3) swamps it by 9 orders of "
                         "magnitude and the mixing weight barely moves -- pass something close to "
                         "the actual diff**2 scale instead (verify empirically per dataset).")
    p.add_argument("--reset-between-epochs", action="store_true")
    p.add_argument("--misalignment-threshold-db", type=float, default=-20.0)
    p.add_argument("--ir-truncate-len", type=int, default=2048)
    p.add_argument("--loop-mode", choices=["open", "closed"], default="open",
                    help="open (default): independent excitation, fast, standard AFC "
                         "misalignment/ERLE benchmarking. closed: the algorithm's own "
                         "output (after --gain-db + saturation) is what's actually "
                         "'played' and acoustically coupled back -- can genuinely "
                         "diverge/howl if gain is high enough, but much slower to run.")
    p.add_argument("--gain-db", type=float, default=0.0,
                    help="Closed loop only: linear gain (dB) applied to e[i] before it's "
                         "'played'. Note the measured IRs here have small raw amplitude "
                         "(not calibrated SPL), so the gain needed to reach instability "
                         "may look unusually large (tens to 100+ dB) -- that's expected.")
    p.add_argument("--saturation-limit", type=float, default=1.0,
                    help="Closed loop only: |played| clip limit, matching a normalized "
                         "[-1,1] full-scale signal (e.g. a real DAC's rails).")
    p.add_argument("--msg-fft-oversample", type=int, default=32,
                    help="fft_len for MSG/ASG = next_pow2(oversample * afl) (default: 32).")
    p.add_argument("--msg-interval-s", type=float, default=None,
                    help="How often to recompute MSG/ASG (default: same as "
                         "--metrics-interval-s, i.e. every snapshot). Must be an integer "
                         "multiple of the metrics interval; the value is zero-order-held "
                         "on the misalignment/ERLE snapshots in between.")
    p.add_argument("--metrics-interval-s", type=float, default=0.05)
    p.add_argument("--n-components-train", type=int, default=None,
                    help="Fix the number of PCA components FIT during training (overrides "
                         "--var-explained's auto-selection). Distinct from --n-components-active, "
                         "which only limits how many of the already-trained components a PCA "
                         "variant may adapt at runtime -- this controls how many get fit in the "
                         "first place. Also raises --device-max-components to at least this value "
                         "if needed, since that cap applies regardless of how K was chosen.")
    p.add_argument("--var-explained", type=float, default=0.95)
    p.add_argument("--device-max-components", type=int, default=16)
    # PEMAFC parameters (48kHz-rescaled defaults, see afc_sim_PLAN.md)
    p.add_argument("--mu", type=float, default=0.03,
                    help="NLMS step size. 0.001 (old default) converges in tens of "
                         "seconds -- far too slow for a hearing aid. 0.03 reaches "
                         "~-15dB misalignment in well under 1s for PEMAFC-A on real "
                         "data (see afc_sim_PLAN.md). Larger mu converges faster but "
                         "settles at a worse steady-state misalignment floor -- this "
                         "value has not been separately validated for the PCA-"
                         "constrained variant, which adapts in a much lower-"
                         "dimensional space and may want a different mu.")
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--ar-order", type=int, default=12)
    p.add_argument("--ar-block-len", type=int, default=640)
    p.add_argument("--ar-reg", type=float, default=1e-5)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--save-wav-dir", type=str, default=None,
                    help="If given, save float32 WAVs (spk/mic/e per algorithm x epoch) "
                         "into this directory. Off by default -- adds up fast at 48kHz.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        "results", datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"Loading conditions from {args.npz_path} ...")
    conds = conditions_mod.load_conditions_from_periods(
        args.npz_path, n_train=args.n_train, n_eval=args.n_eval, seed=args.seed,
        min_quality_percentile=args.min_quality_percentile, n_average=args.n_average)
    print(f"  {len(conds.train_ids)} train periods, {len(conds.eval_ids)} held-out eval periods")
    print(f"  fs={conds.fs:.0f} Hz, mls_length={conds.mls_length}")

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    needs_pca = any("pca" in v for v in variant_names)

    pca_csv_path = None
    pca_fit_info = None
    if needs_pca:
        device_max_components = args.device_max_components
        if args.n_components_train is not None and args.n_components_train > device_max_components:
            device_max_components = args.n_components_train
        pca_csv_path = os.path.join(tempfile.mkdtemp(), "pca_model.csv")
        print(f"\nTraining PCA model (afl={args.afl}) from {len(conds.train_ids)} training conditions ...")
        mean_ir, components, ratio, cumulative, K = pca_training.train_pca_model(
            conds, afl=args.afl, out_path=pca_csv_path, n_components=args.n_components_train,
            var_explained=args.var_explained, device_max_components=device_max_components)
        print(f"  Kept K={K} components (cumulative explained variance "
              f"{cumulative[K-1]*100:.1f}%), wrote {pca_csv_path}")
        pca_fit_info = {
            "n_training_conditions": len(conds.train_ids),
            "K_fitted": K,
            "cumulative_variance_explained": float(cumulative[K - 1]),
            "var_explained_target": args.var_explained,
            "n_components_train_requested": args.n_components_train,
            "device_max_components": device_max_components,
        }

    common_kwargs = dict(mu=args.mu, delta=args.delta, alpha=args.alpha,
                          ar_order=args.ar_order, ar_block_len=args.ar_block_len,
                          ar_reg=args.ar_reg)

    algorithms = []
    for vname in variant_names:
        if vname == "null_afc":
            algorithms.append(create_algorithm(vname, afl=args.afl))
        elif vname in ("pemafc_f", "pemafc_a", "nlms"):
            algorithms.append(create_algorithm(vname, afl=args.afl, **common_kwargs))
        elif vname in ("pemafc_pca_a", "pemafc_pca_f", "nlms_pca"):
            if pca_csv_path is None:
                raise RuntimeError(f"{vname} requires a trained PCA model")
            algorithms.append(create_algorithm(
                vname, pca_model_path=pca_csv_path,
                n_components_active=args.n_components_active, **common_kwargs))
        elif vname == "soft_pca":
            if pca_csv_path is None:
                raise RuntimeError(f"{vname} requires a trained PCA model")
            # One instance per alpha value, each its own named plot line --
            # not one algorithm with a single alpha, since the point is
            # comparing several soft-constraint strengths at once.
            if args.soft_pca_alphas is not None:
                # name_override=None here (not "SoftPCA (alpha=...)") since
                # PEMAFCSoftPCA.name already appends "(alpha=X, K=Y/Z)" itself --
                # passing it here too just duplicated it in every printed name.
                alpha_name_pairs = [(float(a.strip()), None) for a in args.soft_pca_alphas.split(",")]
            else:
                alpha_name_pairs = [(soft_pca_alpha_from_tau(conds.fs, float(t.strip())),
                                     f"SoftPCA (tau={float(t.strip())*1000:.0f}ms)")
                                    for t in args.soft_pca_taus.split(",")]
            for soft_pca_alpha, name_override in alpha_name_pairs:
                algorithms.append(create_algorithm(
                    vname, pca_model_path=pca_csv_path, soft_pca_alpha=soft_pca_alpha,
                    n_components_active=args.n_components_active,
                    name_override=name_override, **common_kwargs))
        elif vname == "blend_pca":
            if pca_csv_path is None:
                raise RuntimeError(f"{vname} requires a trained PCA model")
            for w_s in args.blend_weights.split(","):
                w = float(w_s.strip())
                algorithms.append(create_algorithm(
                    vname, pca_model_path=pca_csv_path, blend_weight=w,
                    n_components_active=args.n_components_active, **common_kwargs))
        elif vname == "blend_pca_dynamic":
            if pca_csv_path is None:
                raise RuntimeError(f"{vname} requires a trained PCA model")
            algorithms.append(create_algorithm(
                vname, pca_model_path=pca_csv_path,
                mu_lambda=args.blend_dynamic_mu_lambda,
                blend_weight_init=args.blend_dynamic_weight_init,
                lambda_reg=args.blend_dynamic_lambda_reg,
                n_components_active=args.n_components_active, **common_kwargs))
        else:
            raise ValueError(f"Unknown algorithm variant '{vname}'")
    print(f"\nAlgorithms: {[a.name for a in algorithms]}")

    epochs = [EpochSpec(condition_id=cid, duration_s=args.epoch_duration)
              for cid in conds.eval_ids]
    print(f"\nEpoch schedule ({len(epochs)} epochs x {args.epoch_duration}s = "
          f"{len(epochs)*args.epoch_duration:.0f}s per algorithm):")
    for e in epochs:
        print(f"  {e.condition_id}  (quality={conds.quality_by_id[e.condition_id]:.1f})")

    sim_config = SimulationConfig(
        fs=conds.fs, block_size=args.block_size,
        forward_signal_spec=SignalSpec(kind=args.signal, seed=args.seed,
                                        amplitude=args.signal_amplitude, wav_path=args.wav_path,
                                        color_pole=args.signal_color_pole,
                                        ar_fit_start_s=args.ar_fit_start_s,
                                        ar_fit_duration_s=args.ar_fit_duration_s,
                                        ar_fit_order=args.ar_fit_order),
        epochs=epochs, reset_between_epochs=args.reset_between_epochs,
        misalignment_threshold_db=args.misalignment_threshold_db,
        loop_mode=args.loop_mode, gain_db=args.gain_db, saturation_limit=args.saturation_limit,
        metrics_interval_s=args.metrics_interval_s, msg_interval_s=args.msg_interval_s,
        msg_fft_oversample=args.msg_fft_oversample)
    if args.loop_mode == "closed":
        print(f"\nClosed loop: gain={args.gain_db:.1f} dB, saturation_limit={args.saturation_limit}")

    feedback_path = FeedbackPath(ir_truncate_len=args.ir_truncate_len)

    print("\nRunning simulation ...")
    if args.save_wav_dir:
        print(f"  Saving spk/mic/e WAVs to {args.save_wav_dir}/")
    results = run_simulation(sim_config, algorithms, conds, feedback_path,
                              save_wav_dir=args.save_wav_dir)

    results_io.save_all(results, sim_config, conds, out_dir, source_npz_path=args.npz_path,
                         cli_args=vars(args), algorithm_names=[a.name for a in algorithms],
                         pca_fit_info=pca_fit_info)
    print(f"\nSaved results to {out_dir}/ (results.csv, manifest.json, "
          f"misalignment_db.png, misalignment_linear.png, erle.png, msg.png, asg.png, "
          f"diagnostics_*.png if applicable)")

    print("\nConvergence time to threshold "
          f"({args.misalignment_threshold_db} dB) / final ASG per epoch:")
    for r in results:
        ct = f"{r.convergence_time_s:.2f}s" if r.convergence_time_s is not None else "not reached"
        sat = f", saturated {r.saturated_fraction*100:.1f}% of samples" if r.saturated_fraction else ""
        final_asg = r.asg_db[-1] if len(r.asg_db) else float("nan")
        print(f"  {r.algorithm_name:30s} epoch {r.epoch_index} ({r.condition_id}): {ct}{sat}, "
              f"ASG={final_asg:.1f}dB (MSG no-AFC={r.msg_without_afc_db:.1f}dB)")


if __name__ == "__main__":
    main()
