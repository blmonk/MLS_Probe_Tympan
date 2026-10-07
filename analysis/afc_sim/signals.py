"""Excitation/forward-signal generators for driving the open-loop simulation.
Registry-style so a future signal type (e.g. speech-shaped noise) is a
one-line addition, matching the extensibility goal for the rest of afc_sim."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfiltfilt

from .algorithms.whitening import biased_autocorr, levinson_durbin


@dataclass
class SignalSpec:
    kind: Literal["white", "pink", "colored", "wav", "ar_fit"] = "white"
    seed: int = 0
    amplitude: float = 0.1     # RMS-ish target amplitude (excitation is open-loop; keep well
                                # under 1.0 so the simulated mic signal has headroom)
    wav_path: str | None = None   # required if kind == "wav" or "ar_fit"
    color_pole: float = 0.9    # kind == "colored" only: AR(1) pole, x[n] = white[n] + pole*x[n-1].
                                # Deliberately NOT reusing PEMAFCMode.F's `alpha` name/plumbing --
                                # this is a property of the TEST SIGNAL (what coloration the
                                # excitation has), independent of any algorithm's own assumption
                                # about it, even when a test run intentionally sets them equal to
                                # exercise PEMAFC-F's designed-for case. 0.9 matches PEMAFCFull's
                                # own default `alpha`, which is exactly the point: this signal is
                                # the coloration mode F's fixed pre-emphasis [1, -alpha] is designed
                                # to invert (see afc_sim/tests or the gain_amplitude_sweep.py-era
                                # conversation record for why white noise structurally disadvantages
                                # mode F and this does the opposite).
    ar_fit_start_s: float = 0.0     # kind == "ar_fit" only: where in wav_path to fit the AR model from
    ar_fit_duration_s: float = 5.0  # kind == "ar_fit" only: length of that fit segment
    ar_fit_order: int = 20          # kind == "ar_fit" only: LPC order for the fitted envelope


def _load_wav_mono(path: str, fs: float) -> np.ndarray:
    wav_fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if wav_fs != fs:
        raise ValueError(f"WAV sample rate {wav_fs} != simulation fs {fs} "
                          f"(resampling not implemented -- provide a matching-rate file)")
    return data


def generate_excitation(spec: SignalSpec, n_samples: int, fs: float) -> np.ndarray:
    if spec.kind == "white":
        rng = np.random.default_rng(spec.seed)
        x = rng.normal(size=n_samples)
    elif spec.kind == "pink":
        rng = np.random.default_rng(spec.seed)
        white = rng.normal(size=n_samples)
        # cheap pink approximation: integrate white noise then remove drift
        # with a highpass (true 1/f pink noise isn't needed for AFC excitation
        # purposes -- any reasonably broadband, non-flat-spectrum probe works)
        sos = butter(2, 5.0, btype="highpass", fs=fs, output="sos")
        x = sosfiltfilt(sos, np.cumsum(white))
    elif spec.kind == "colored":
        # AR(1): x[n] = white[n] + color_pole*x[n-1] -- the exact inverse of
        # PEMAFCFull mode F's fixed pre-emphasis y_f[n] = y[n] - alpha*y[n-1]
        # when color_pole == alpha, so this signal is precisely what that
        # fixed whitening filter is designed to flatten (unlike white noise,
        # which it has nothing to do on, or "pink", whose spectral shape
        # doesn't match mode F's single-pole assumption).
        rng = np.random.default_rng(spec.seed)
        white = rng.normal(size=n_samples)
        x = lfilter([1.0], [1.0, -spec.color_pole], white)
    elif spec.kind == "wav":
        if not spec.wav_path:
            raise ValueError("SignalSpec.wav_path is required when kind=='wav'")
        data = _load_wav_mono(spec.wav_path, fs)
        if len(data) < n_samples:
            reps = int(np.ceil(n_samples / len(data)))
            data = np.tile(data, reps)
        x = data[:n_samples]
    elif spec.kind == "ar_fit":
        # Fit an AR envelope to a real segment (e.g. real speech), then
        # generate STATIONARY noise with that same long-term spectral shape --
        # "speech-shaped noise": same average coloration as real speech
        # (so it still exercises mode F's pre-emphasis / tests whether
        # whitening helps), but none of real speech's pauses and wide dynamic
        # range, which otherwise make the adaptation curves noisy and hard to
        # read. Reuses the project's own whitening.biased_autocorr/
        # levinson_durbin rather than a separate LPC implementation.
        if not spec.wav_path:
            raise ValueError("SignalSpec.wav_path is required when kind=='ar_fit'")
        data = _load_wav_mono(spec.wav_path, fs)
        lo = int(round(spec.ar_fit_start_s * fs))
        hi = lo + int(round(spec.ar_fit_duration_s * fs))
        if hi > len(data):
            raise ValueError(f"ar_fit segment [{spec.ar_fit_start_s}, "
                              f"{spec.ar_fit_start_s + spec.ar_fit_duration_s}]s exceeds "
                              f"{spec.wav_path}'s duration ({len(data)/fs:.2f}s)")
        segment = data[lo:hi] - data[lo:hi].mean()
        r = biased_autocorr(segment, spec.ar_fit_order)
        ar_coeffs, stable = levinson_durbin(r, spec.ar_fit_order)
        if not stable:
            raise RuntimeError(
                f"AR({spec.ar_fit_order}) fit to {spec.wav_path}[{spec.ar_fit_start_s}:"
                f"{spec.ar_fit_start_s + spec.ar_fit_duration_s}]s was unstable "
                f"(|reflection coeff|>=0.999) -- try a different segment or a lower ar_fit_order")
        rng = np.random.default_rng(spec.seed)
        white = rng.normal(size=n_samples)
        # Generation filter is the INVERSE of the whitening filter: whitening
        # applies [1, *ar_coeffs] to flatten a signal with this envelope;
        # generating one from white noise applies the reciprocal, 1/[1, *ar_coeffs].
        x = lfilter([1.0], np.concatenate([[1.0], ar_coeffs]), white)
    else:
        raise ValueError(f"Unknown SignalSpec.kind '{spec.kind}'")

    rms = np.sqrt(np.mean(x**2)) + 1e-15
    return x * (spec.amplitude / rms)
