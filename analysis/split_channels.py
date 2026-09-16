#!/usr/bin/env python3
"""
split_channels.py  —  Split a 2-channel WAV into two mono WAV files.

Usage:
  python split_channels.py <recording.wav>

Outputs:
  <recording>_ch0_mic.wav   — channel 0 (microphone)
  <recording>_ch1_mls.wav   — channel 1 (MLS reference)
"""

import sys
import pathlib
import numpy as np

try:
    import scipy.io.wavfile as wavfile
    def load_wav(path):
        return wavfile.read(path)
    def save_wav(path, fs, data):
        wavfile.write(path, fs, data)
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
    def save_wav(path, fs, data):
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(data.dtype.itemsize)
            wf.setframerate(fs)
            wf.writeframes(data.tobytes())

if len(sys.argv) < 2:
    sys.exit("Usage: python split_channels.py <recording.wav>")

src = pathlib.Path(sys.argv[1])
if not src.exists():
    sys.exit(f"ERROR: File not found: {src}")

fs, data = load_wav(str(src))

if data.ndim == 1:
    sys.exit("ERROR: WAV is already mono.")
if data.shape[1] < 2:
    sys.exit(f"ERROR: WAV has only {data.shape[1]} channel(s); need at least 2.")

print(f"Loaded : {src}")
print(f"  Sample rate : {fs} Hz")
print(f"  Duration    : {data.shape[0]/fs:.3f} s  ({data.shape[0]} samples)")
print(f"  Channels    : {data.shape[1]}")
print(f"  Data type   : {data.dtype}")

stem = src.stem
out_mic = src.with_name(f"{stem}_ch0_mic.wav")
out_mls = src.with_name(f"{stem}_ch1_mls.wav")

save_wav(str(out_mic), fs, data[:, 0])
save_wav(str(out_mls), fs, data[:, 1])

print(f"\nWrote: {out_mic}")
print(f"Wrote: {out_mls}")
