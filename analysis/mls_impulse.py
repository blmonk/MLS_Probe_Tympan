#!/usr/bin/env python3
"""
mls_impulse.py  —  Reconstruct impulse responses from a 2-channel MLS probe WAV file.

The WAV file must come from MLS_Probe_Tympan (or compatible):
  Channel 0 (left):  microphone signal
  Channel 1 (right): MLS reference signal

Algorithm:
  1. Regenerate the exact MLS used by the hardware (same LFSR parameters) for
     the length you specify with --mls_length.
  2. Cross-correlate the reference channel with the clean MLS to find the period
     phase offset at the start of the recording.
  3. For each collected period: h = IFFT( FFT(mic) * conj(FFT(mls)) ) / N
     (circular cross-correlation, exact for signals shorter than one MLS period).
  4. Plot individual impulse responses stacked, overlaid, and averaged.

Usage:
  python mls_impulse.py <recording.wav> --mls_length 1023
  python mls_impulse.py <recording.wav> --mls_length 8191 --n_impulses 8 --plot_ms 5.0 --skip 3

--mls_length must match whatever length the firmware was actually set to
when the recording was made (see the 'len <val>' serial command and the 'g'
settings printout in MLS_Probe_Tympan). There is no default — a wrong guess
here would silently desync and produce meaningless impulse responses, so the
script always requires it to be stated explicitly.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── WAV loading + filtering (scipy preferred, stdlib fallback) ───────────────
try:
    import scipy.io.wavfile as _wavfile
    from scipy.signal import butter, sosfiltfilt
    def load_wav(path):
        fs, data = _wavfile.read(path)
        return fs, data
    def highpass_filter(signal, cutoff_hz, fs):
        sos = butter(4, cutoff_hz, btype='highpass', fs=fs, output='sos')
        return sosfiltfilt(sos, signal)
except ImportError:
    import wave, struct
    def load_wav(path):
        with wave.open(path, 'rb') as wf:
            fs       = wf.getframerate()
            n_ch     = wf.getnchannels()
            n_frames = wf.getnframes()
            width    = wf.getsampwidth()
            raw      = wf.readframes(n_frames)
        fmt  = {1: 'b', 2: '<i2', 4: '<i4'}[width]
        data = np.frombuffer(raw, dtype=np.dtype(fmt)).reshape(-1, n_ch)
        return fs, data
    def highpass_filter(signal, cutoff_hz, fs):
        sys.exit("ERROR: scipy is required for --highpass filtering. Install with: pip install scipy")


# ── MLS parameters (must match AudioSynthMLS_F32.h's kVariants table) ───────
# Each entry: length -> (degree, tap_mask). tap_mask bit i set => LFSR bit i
# is XORed into the feedback (bit 0 -- the output/LSB -- is always included).
# Brute-force verified: from an all-ones seed, each configuration below
# visits all 2^n - 1 nonzero states before returning to the seed (i.e. it's a
# true maximal-length sequence). Degrees 12, 13, 14 and 16 need 4 taps; no
# 2-tap (trinomial) feedback exists for those degrees.
MLS_VARIANTS = {
    1023:  (10, 0x0081),
    2047:  (11, 0x0005),
    4095:  (12, 0x0107),
    8191:  (13, 0x0027),
    16383: (14, 0x1007),
    32767: (15, 0x0003),
    65535: (16, 0x100B),
}


def generate_mls(length):
    """
    Reproduce AudioSynthMLS_F32::generateSequence() exactly for `length`.
    LFSR structure: register bits r[0..n-1], r[0] = LSB = output.
        feedback  = XOR-parity(state & tap_mask)
        new_state = (state >> 1) | (feedback << (n-1))
    Seed = all-ones (state == length, since length == 2^n - 1).
    Returns float64 array of ±1 values, length `length`.
    """
    degree, tap_mask = MLS_VARIANTS[length]
    lfsr = length  # all-ones seed
    mls  = np.empty(length, dtype=np.float64)
    for i in range(length):
        out      = lfsr & 1
        feedback = bin(lfsr & tap_mask).count("1") & 1
        lfsr     = (lfsr >> 1) | (feedback << (degree - 1))
        mls[i]   = 1.0 if out else -1.0
    return mls


def find_mls_offset(ref, mls_clean):
    """
    Find the sample index in `ref` where the first aligned MLS period begins.

    Cross-correlates the first 4 MLS periods of `ref` against the clean ±1
    MLS template.  The argmax of the correlation is the start of the first
    period that is aligned with our template — i.e. where ref[offset:offset+N]
    matches mls_clean[0:N].

    Returns the integer sample offset.
    """
    N      = len(mls_clean)
    search = ref[: 4 * N].astype(np.float64)
    std    = np.std(search)
    if std < 1e-10:
        raise ValueError(
            "Reference channel appears silent — was MLS playback running during recording?"
        )
    # Normalise both signals so amplitude differences don't affect the peak location
    template = mls_clean / np.std(mls_clean)
    search  /= std
    xcorr    = np.correlate(search, template, mode='valid')
    best     = int(np.argmax(np.abs(xcorr)))

    # Sanity check: a correctly-matched MLS length produces one sharp,
    # dominant correlation peak (MLS autocorrelation is impulse-like). An
    # unremarkable/ambiguous peak here usually means --mls_length doesn't
    # match what the firmware actually played for this recording.
    peak_mag    = abs(xcorr[best])
    typical_mag = np.median(np.abs(xcorr)) + 1e-12
    peak_ratio  = peak_mag / typical_mag
    if peak_ratio < 5.0:
        print(f"WARNING: Sync peak is not sharply dominant (peak/typical ratio "
              f"{peak_ratio:.1f}). Double-check that --mls_length {N} actually "
              f"matches what the firmware used for this recording.")

    return best


def compute_ir(mic_segment, mls_clean):
    """
    Estimate the impulse response from one MLS period of microphone data via
    circular cross-correlation in the frequency domain:

        h[n] = IFFT( FFT(mic)[k] * conj(FFT(mls))[k] ) / energy(mls)

    For a ±1 MLS of length N, energy = N exactly.  This estimate is accurate
    when the true impulse response is shorter than one MLS period.
    """
    N   = len(mls_clean)
    MIC = np.fft.rfft(mic_segment.astype(np.float64), n=N)
    MLS = np.fft.rfft(mls_clean, n=N)
    h   = np.fft.irfft(MIC * np.conj(MLS), n=N)
    return h / np.dot(mls_clean, mls_clean)   # divide by N


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("wav_file",
                        help="2-channel WAV file (ch0=mic, ch1=MLS reference)")
    parser.add_argument("--mls_length", type=int, required=True, choices=sorted(MLS_VARIANTS),
                        help="MLS length the firmware used for this recording (no default — "
                             "must match exactly, see the module docstring)")
    parser.add_argument("--n_impulses", type=int,   default=5,
                        help="Number of impulse responses to collect and plot (default: 5)")
    parser.add_argument("--plot_ms",    type=float,  default=None,
                        help="Time window to display in ms (default: full MLS period)")
    parser.add_argument("--skip",       type=int,    default=2,
                        help="Periods to discard after sync point before collecting (default: 2)")
    parser.add_argument("--highpass",   type=float,  default=None,
                        help="Apply a 4th-order zero-phase Butterworth high-pass filter to the "
                             "mic channel before computing IRs, e.g. --highpass 100")
    args = parser.parse_args()

    mls_length = args.mls_length

    # ── Load WAV ──────────────────────────────────────────────────────────────
    try:
        fs, data = load_wav(args.wav_file)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {args.wav_file}")
    except Exception as e:
        sys.exit(f"ERROR loading WAV: {e}")

    if data.ndim == 1:
        sys.exit("ERROR: WAV is mono — expected 2 channels (ch0=mic, ch1=MLS reference).")
    if data.shape[1] < 2:
        sys.exit(f"ERROR: WAV has only {data.shape[1]} channel(s); need at least 2.")

    print(f"Loaded  : {args.wav_file}")
    print(f"  Sample rate : {fs} Hz")
    print(f"  Duration    : {data.shape[0] / fs:.3f} s  ({data.shape[0]} samples)")
    print(f"  Data type   : {data.dtype}")
    print(f"  Channels    : {data.shape[1]}")

    # Normalise to float64 [-1, 1]
    if   np.issubdtype(data.dtype, np.integer):
        info  = np.iinfo(data.dtype)
        data  = data.astype(np.float64) / max(abs(info.min), abs(info.max))
    else:
        data  = data.astype(np.float64)

    mic = data[:, 0]   # left  channel — microphone
    ref = data[:, 1]   # right channel — MLS reference

    # ── High-pass filter ──────────────────────────────────────────────────────
    if args.highpass is not None:
        nyq = fs / 2.0
        if args.highpass <= 0 or args.highpass >= nyq:
            sys.exit(f"ERROR: --highpass cutoff must be between 0 and {nyq:.0f} Hz.")
        mic = highpass_filter(mic, args.highpass, fs)
        print(f"\nHighpass : 4th-order Butterworth at {args.highpass:.1f} Hz (zero-phase)")

    period_ms = 1000.0 * mls_length / fs
    print(f"\nMLS      : {mls_length} samples = {period_ms:.2f} ms per period "
          f"({fs / mls_length:.1f} Hz repetition rate)")

    # ── Generate clean MLS ────────────────────────────────────────────────────
    mls = generate_mls(mls_length)

    # ── Synchronise ───────────────────────────────────────────────────────────
    if len(ref) < 4 * mls_length + args.skip * mls_length + args.n_impulses * mls_length:
        print("WARNING: Recording may be too short for the requested parameters.")
    if len(ref) < 4 * mls_length:
        sys.exit("ERROR: Recording is too short to synchronise (need ≥ 4 MLS periods).")

    try:
        sync_offset = find_mls_offset(ref, mls)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    print(f"  Sync offset : sample {sync_offset}  ({1000.0 * sync_offset / fs:.2f} ms into first period)")

    collect_start = sync_offset + args.skip * mls_length
    n_available   = max(0, (len(mic) - collect_start) // mls_length)
    n_collect     = min(args.n_impulses, n_available)

    print(f"  Skipping {args.skip} periods after sync; {n_available} complete periods available")

    if n_collect < 1:
        sys.exit(
            "ERROR: Not enough data to collect any impulse responses. "
            "Try reducing --skip or --n_impulses, or use a longer recording."
        )
    if n_collect < args.n_impulses:
        print(f"WARNING: Only {n_collect} periods available (requested {args.n_impulses}).")

    # ── Compute impulse responses ─────────────────────────────────────────────
    IRs = np.empty((n_collect, mls_length), dtype=np.float64)
    for k in range(n_collect):
        s        = collect_start + k * mls_length
        IRs[k]   = compute_ir(mic[s : s + mls_length], mls)

    mean_IR = IRs.mean(axis=0)

    # ── Time axis ─────────────────────────────────────────────────────────────
    t_ms   = 1000.0 * np.arange(mls_length) / fs
    n_plot = min(mls_length, int(args.plot_ms * fs / 1000) if args.plot_ms else mls_length)
    t_show = t_ms[:n_plot]

    # ── Console summary ───────────────────────────────────────────────────────
    peak_idx = np.argmax(np.abs(IRs), axis=1)
    print(f"\nPeak delay per IR (ms): {np.round(1000.0 * peak_idx / fs, 3).tolist()}")
    print(f"Mean peak delay        : {1000.0 * np.mean(peak_idx) / fs:.3f} ms")

    # ── Figure 1: stacked individual IRs ─────────────────────────────────────
    fig1, axes = plt.subplots(
        n_collect, 1,
        figsize=(11, 2.4 * n_collect),
        sharex=True,
    )
    if n_collect == 1:
        axes = [axes]

    for k, (ax, h) in enumerate(zip(axes, IRs)):
        ax.plot(t_show, h[:n_plot], lw=0.8, color="steelblue")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(f"IR #{args.skip + k + 1}", fontsize=9)
        ax.grid(True, alpha=0.3)
        # mark peak
        pk = int(np.argmax(np.abs(h[:n_plot])))
        ax.axvline(t_show[pk], color="tomato", lw=0.8, ls="--", alpha=0.7)

    axes[-1].set_xlabel("Time (ms)")
    hp_tag = f", HP {args.highpass:.0f} Hz" if args.highpass else ""
    fig1.suptitle(f"MLS Impulse Responses (individual{hp_tag}) — {args.wav_file}", fontsize=11)
    plt.tight_layout()

    # ── Figure 2: overlaid ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_collect))
    for k, (h, c) in enumerate(zip(IRs, colors)):
        ax2.plot(t_show, h[:n_plot], lw=0.8, alpha=0.75,
                 color=c, label=f"IR #{args.skip + k + 1}")
    ax2.plot(t_show, mean_IR[:n_plot], lw=1.6, color="black",
             ls="--", label="Mean", zorder=5)
    ax2.axhline(0, color="k", lw=0.4)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Impulse Responses Overlaid{hp_tag} — {args.wav_file}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── Figure 3: mean IR ─────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    ax3.plot(t_show, mean_IR[:n_plot], lw=1.0, color="steelblue")
    ax3.axhline(0, color="k", lw=0.4)
    pk_mean = int(np.argmax(np.abs(mean_IR[:n_plot])))
    ax3.axvline(t_show[pk_mean], color="tomato", lw=1.0, ls="--",
                label=f"Peak @ {t_show[pk_mean]:.3f} ms")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title(f"Mean Impulse Response (averaged over {n_collect} periods{hp_tag}) — {args.wav_file}")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
