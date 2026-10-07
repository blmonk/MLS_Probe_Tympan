#!/usr/bin/env python3
"""
gain_amplitude_sweep.py -- Maps the usable (signal_amplitude, gain_db) region
for closed-loop simulation: too little gain relative to the true IR's tiny
raw amplitude means insufficient excitation of the real feedback path (the
algorithm can't learn anything real, efbp is pure adaptation noise); too much
gain saturates immediately regardless of any real acoustic loop dynamics.
Neither extreme demonstrates genuine loop stability/instability -- this
script finds the useful middle range for a given condition/algorithm.

Usage:
  python -m afc_sim.gain_amplitude_sweep ../data/self/AUDIO001_ir_hp100.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from afc_sim import conditions as conditions_mod
from afc_sim.algorithms.pemafc import PEMAFCFull
from afc_sim.algorithms.whitening import PEMAFCMode
from afc_sim.closed_loop import run_closed_loop_epoch
from afc_sim.conditions import zero_lag_window
from afc_sim.feedback_path import FeedbackPath
from afc_sim.signals import SignalSpec, generate_excitation
from afc_sim import metrics


def run_sweep(npz_path, afl=128, duration_s=1.0, ar_block_len=160, mu=0.001,
              amplitudes=(0.001, 0.003, 0.01, 0.03, 0.1),
              gains_db=(20, 30, 40, 50, 60, 70, 80, 90, 100), seed=0):
    conds = conditions_mod.load_conditions_from_periods(
        npz_path, n_train=10, n_eval=1, seed=seed, min_quality_percentile=50.0)
    cid = conds.eval_ids[0]
    raw_ir = conds.ir_by_id[cid]
    fs = conds.fs
    n = int(round(duration_s * fs))

    fp = FeedbackPath(ir_truncate_len=2048)
    fp.set_ir(raw_ir)
    true_ir_full = fp.active_ir
    true_ir_afl = zero_lag_window(raw_ir, afl, lag=0)  # closed loop -- see zero_lag_window docstring

    results = []
    for amp in amplitudes:
        spec = SignalSpec(kind="white", seed=seed, amplitude=amp)
        desired = generate_excitation(spec, n, fs)
        for gain_db in gains_db:
            gain_lin = 10 ** (gain_db / 20.0)
            algo = PEMAFCFull(afl=afl, mu=mu, delta=1e-3, mode=PEMAFCMode.F, alpha=0.9,
                               ar_order=12, ar_block_len=ar_block_len)
            mic, played, e, sat = run_closed_loop_epoch(
                algo, desired, true_ir_full, gain_linear=gain_lin)
            mis = metrics.misalignment_db(algo.get_estimated_ir(), true_ir_afl)
            results.append({
                "amplitude": amp, "gain_db": gain_db,
                "saturated_pct": 100 * sat, "misalignment_db": mis,
                "max_played": float(np.max(np.abs(played))),
            })
    return results, cid


def classify(r):
    if r["saturated_pct"] > 5.0:
        return "SATURATING"
    if r["misalignment_db"] > 15.0:
        return "UNDER-EXCITED"
    return "usable"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_path")
    p.add_argument("--afl", type=int, default=128)
    p.add_argument("--duration-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results, cid = run_sweep(args.npz_path, afl=args.afl, duration_s=args.duration_s, seed=args.seed)
    print(f"Condition: {cid}, afl={args.afl}, duration={args.duration_s}s\n")

    amps = sorted(set(r["amplitude"] for r in results))
    gains = sorted(set(r["gain_db"] for r in results))
    by_key = {(r["amplitude"], r["gain_db"]): r for r in results}

    header = "amplitude \\ gain_db  " + "  ".join(f"{g:>6d}" for g in gains)
    print(header)
    for amp in amps:
        row = f"{amp:<20.4f} "
        for g in gains:
            r = by_key[(amp, g)]
            tag = classify(r)
            symbol = {"SATURATING": " SAT ", "UNDER-EXCITED": " weak", "usable": f"{r['misalignment_db']:5.1f}"}[tag]
            row += f"  {symbol:>6s}"
        print(row)

    print("\nLegend: number = final misalignment (dB, lower better) in the usable region; "
          "'SAT' = >5% samples saturated (clipping-dominated, not real loop dynamics); "
          "'weak' = misalignment >15dB with no saturation (feedback signal too weak to learn from).")

    usable = [r for r in results if classify(r) == "usable"]
    if usable:
        best = min(usable, key=lambda r: r["misalignment_db"])
        print(f"\nBest usable point: amplitude={best['amplitude']}, gain_db={best['gain_db']} "
              f"-> misalignment={best['misalignment_db']:.2f} dB, saturated={best['saturated_pct']:.1f}%")


if __name__ == "__main__":
    main()
