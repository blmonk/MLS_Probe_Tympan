"""
Main simulation loop: for a schedule of epochs (each pinning one held-out
condition's impulse response active for a fixed duration), drive every
algorithm variant with the SAME excitation realization per epoch (fair
comparison), recording misalignment/ERLE time series.

Reset semantics: reset() is called once per algorithm at the very start of
the whole run, NOT between epochs, unless reset_between_epochs=True -- the
default measures real reconvergence behavior after a feedback-path switch
("let it converge before switching to the next impulse"), which is the
scientifically interesting question here.
"""

import os
import re
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import numpy as np
from scipy.io import wavfile

from . import metrics
from .algorithms.base import AFCAlgorithm
from .closed_loop import run_closed_loop_epoch
from .conditions import ConditionSet, zero_lag_window
from .feedback_path import FeedbackPath
from .signals import SignalSpec, generate_excitation


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


@dataclass
class EpochSpec:
    condition_id: str
    duration_s: float


@dataclass
class SimulationConfig:
    fs: float
    block_size: int
    forward_signal_spec: SignalSpec
    epochs: list[EpochSpec]
    metrics_interval_s: float = 0.05
    reset_between_epochs: bool = False
    misalignment_threshold_db: float = -20.0
    loop_mode: Literal["open", "closed"] = "open"
    gain_db: float = 0.0             # closed loop only: applied to e[i] before it's "played"
    saturation_limit: float = 1.0    # closed loop only: |played| clip limit
    msg_interval_s: Optional[float] = None   # None => same as metrics_interval_s. Must be an
                                              # integer multiple of metrics_interval_s -- MSG/ASG
                                              # share the same `t` axis as misalignment/ERLE
                                              # (zero-order-held between recomputes), not a second one.
    msg_fft_oversample: int = 32             # fft_len = next_pow2(oversample * afl)


@dataclass
class TrialResult:
    algorithm_name: str
    epoch_index: int
    condition_id: str
    t: np.ndarray
    misalignment_db: np.ndarray
    erle_db: np.ndarray
    convergence_time_s: Optional[float]
    saturated_fraction: Optional[float] = None  # closed loop only: fraction of samples
                                                 # clipped at saturation_limit ("howling")
    msg_with_afc_db: np.ndarray = None          # aligned to the same `t` as the other metrics
    msg_without_afc_db: float = None            # constant per (algo, epoch) -- depends on
                                                 # (true_ir_afl, algo.afl), not just condition_id
    asg_db: np.ndarray = None                   # msg_with_afc_db - msg_without_afc_db
    diagnostics: dict = field(default_factory=dict)  # algo.get_diagnostics() series, aligned to `t`;
                                                       # empty for algorithms that report nothing extra


def _generate_epoch_excitations(config: SimulationConfig) -> list[np.ndarray]:
    excitations = []
    for i, epoch in enumerate(config.epochs):
        n = int(round(epoch.duration_s * config.fs))
        spec = replace(config.forward_signal_spec, seed=config.forward_signal_spec.seed + i)
        excitations.append(generate_excitation(spec, n, config.fs))
    return excitations


def run_simulation(config: SimulationConfig, algorithms: list[AFCAlgorithm],
                    conditions: ConditionSet, feedback_path: FeedbackPath,
                    save_wav_dir: str = None) -> list[TrialResult]:
    """
    save_wav_dir: if given, writes float32 WAVs per (algorithm, epoch) --
    <dir>/<algo>_epoch<N>_<condition_id>_{spk,mic,e}.wav -- spk is the
    loudspeaker/excitation signal, mic is the feedback-contaminated
    microphone signal FeedbackPath produced, e is that algorithm's residual
    output after cancellation. Off by default since a full run's worth of
    audio at 48kHz adds up quickly.
    """
    epoch_excitations = _generate_epoch_excitations(config)
    metrics_interval_samples = max(1, int(round(config.metrics_interval_s * config.fs)))
    results: list[TrialResult] = []
    if save_wav_dir:
        os.makedirs(save_wav_dir, exist_ok=True)

    gain_linear = 10 ** (config.gain_db / 20.0)

    for algo in algorithms:
        algo.reset()
        t_offset = 0.0

        for epoch_idx, (epoch, spk) in enumerate(zip(config.epochs, epoch_excitations)):
            if config.reset_between_epochs and epoch_idx > 0:
                algo.reset()

            raw_ir = conditions.ir_by_id[epoch.condition_id]
            # lag=0 for closed loop: closed_loop.py's own convolution already
            # bakes in the minimum causal lag on this same array, so no
            # further shift is correct here -- see zero_lag_window's docstring.
            lag = 0 if config.loop_mode == "closed" else 1
            true_ir_afl = zero_lag_window(raw_ir, algo.afl, lag=lag)
            n = len(spk)

            acc = _MetricsAccumulator(algo, true_ir_afl, config.fs, t_offset, metrics_interval_samples,
                                       msg_interval_s=config.msg_interval_s,
                                       metrics_interval_s=config.metrics_interval_s,
                                       msg_fft_oversample=config.msg_fft_oversample)
            saturated_fraction = None

            if config.loop_mode == "open":
                feedback_path.set_ir(raw_ir)
                mic = feedback_path.contaminate(spk)
                e_full = np.empty(n) if save_wav_dir else None
                pos = 0
                while pos < n:
                    bs = min(config.block_size, n - pos)
                    mic_blk = mic[pos:pos + bs]
                    spk_blk = spk[pos:pos + bs]
                    e_blk = algo.process_block(mic_blk, spk_blk)
                    if not np.all(np.isfinite(algo.get_estimated_ir())) or not np.all(np.isfinite(e_blk)):
                        raise RuntimeError(
                            f"{algo.name} diverged (non-finite state) at epoch {epoch_idx} "
                            f"({epoch.condition_id}), sample {pos}. Aborting this trial.")
                    if e_full is not None:
                        e_full[pos:pos + bs] = e_blk
                    acc.feed(pos + bs, mic_blk, e_blk)
                    pos += bs
                played = spk  # the "loudspeaker signal" IS the excitation in open loop
            else:
                feedback_path.set_ir(raw_ir)  # unused for contamination, kept in sync for consistency
                true_ir_full = feedback_path.active_ir
                mic, played, e_full_computed, saturated_fraction = run_closed_loop_epoch(
                    algo, desired=spk, true_ir=true_ir_full, gain_linear=gain_linear,
                    saturation_limit=config.saturation_limit,
                    on_chunk=lambda pos, mic_chunk, e_chunk: acc.feed(pos, mic_chunk, e_chunk))
                e_full = e_full_computed if save_wav_dir else None

            t_arr, mis_arr, erle_arr, msg_with_arr, msg_without, asg_arr, diagnostics = acc.finalize()
            conv_t = metrics.convergence_time_s(
                t_arr - t_offset, mis_arr, config.misalignment_threshold_db)

            if save_wav_dir:
                stem = f"{_safe_filename(algo.name)}_epoch{epoch_idx}_{epoch.condition_id}"
                fs_int = int(round(config.fs))
                wavfile.write(os.path.join(save_wav_dir, f"{stem}_spk.wav"), fs_int, played.astype(np.float32))
                wavfile.write(os.path.join(save_wav_dir, f"{stem}_mic.wav"), fs_int, mic.astype(np.float32))
                wavfile.write(os.path.join(save_wav_dir, f"{stem}_e.wav"), fs_int, e_full.astype(np.float32))

            results.append(TrialResult(
                algorithm_name=algo.name, epoch_index=epoch_idx, condition_id=epoch.condition_id,
                t=t_arr, misalignment_db=mis_arr, erle_db=erle_arr, convergence_time_s=conv_t,
                saturated_fraction=saturated_fraction, msg_with_afc_db=msg_with_arr,
                msg_without_afc_db=msg_without, asg_db=asg_arr, diagnostics=diagnostics))

            t_offset += epoch.duration_s

    return results


class _MetricsAccumulator:
    """Shared misalignment/ERLE/MSG/ASG snapshot logic for both loop modes --
    fed sample-position checkpoints as each mode's own loop produces them,
    so the snapshot cadence (metrics_interval_s) is identical regardless of
    whether the underlying processing granularity is block_size (open loop,
    batched for speed) or one sample at a time (closed loop, a genuine
    per-sample causal loop -- see closed_loop.py).

    MSG/ASG share this same `t` axis rather than getting their own: MSG_with_AFC
    and ASG are only recomputed every `msg_interval_s` (an integer multiple of
    metrics_interval_s, since the FFT-based crossing search is more work than
    misalignment/ERLE) and zero-order-held (repeated) on the snapshots in
    between -- keeps TrialResult/results_io.py's "one shared t per trial"
    long-format-CSV assumption intact. MSG_without_AFC is computed once here
    (constant per epoch, doesn't depend on efbp)."""

    def __init__(self, algo, true_ir_afl, fs, t_offset, metrics_interval_samples,
                 msg_interval_s=None, metrics_interval_s=0.05, msg_fft_oversample=32):
        self.algo = algo
        self.true_ir_afl = true_ir_afl
        self.fs = fs
        self.t_offset = t_offset
        self.metrics_interval_samples = metrics_interval_samples
        self.t_list, self.mis_list, self.erle_list = [], [], []
        self.msg_with_list, self.asg_list = [], []
        self.diag_lists: dict[str, list] = {}  # lazily populated from algo.get_diagnostics()
        self.next_metrics_at = 0
        self._mic_buf, self._e_buf = [], []

        self._fft_len = metrics.default_msg_fft_len(algo.afl, msg_fft_oversample)
        self.msg_without_afc_db = metrics.msg_db(true_ir_afl, self._fft_len, fs=fs)

        msg_interval_s = metrics_interval_s if msg_interval_s is None else msg_interval_s
        self._msg_every_n = max(1, round(msg_interval_s / metrics_interval_s))
        self._snapshot_count = 0
        self._last_msg_with = self.msg_without_afc_db  # efbp starts at/near zero -> residual ~= true_ir
        self._last_asg = 0.0

    def feed(self, pos: int, mic_chunk: np.ndarray, e_chunk: np.ndarray) -> None:
        self._mic_buf.append(mic_chunk)
        self._e_buf.append(e_chunk)
        if pos >= self.next_metrics_at:
            estimated_ir = self.algo.get_estimated_ir()
            self.t_list.append(self.t_offset + pos / self.fs)
            self.mis_list.append(metrics.misalignment_db(estimated_ir, self.true_ir_afl))
            self.erle_list.append(metrics.erle_db(
                np.concatenate(self._mic_buf), np.concatenate(self._e_buf)))

            if self._snapshot_count % self._msg_every_n == 0:
                residual = self.true_ir_afl - estimated_ir
                self._last_msg_with = metrics.msg_db(residual, self._fft_len, fs=self.fs)
                self._last_asg = metrics.asg_db(self._last_msg_with, self.msg_without_afc_db)
            self.msg_with_list.append(self._last_msg_with)
            self.asg_list.append(self._last_asg)

            for key, val in self.algo.get_diagnostics().items():
                self.diag_lists.setdefault(key, []).append(val)

            self._mic_buf, self._e_buf = [], []
            self.next_metrics_at += self.metrics_interval_samples
            self._snapshot_count += 1

    def finalize(self):
        diagnostics = {k: np.array(v) for k, v in self.diag_lists.items()}
        return (np.array(self.t_list), np.array(self.mis_list), np.array(self.erle_list),
                np.array(self.msg_with_list), self.msg_without_afc_db, np.array(self.asg_list),
                diagnostics)
