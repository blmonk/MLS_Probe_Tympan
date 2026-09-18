# MLS_Probe_Tympan

Arduino sketch for the [Tympan](https://www.tympan.org/) RevE + EarpieceShield
that plays a Maximum Length Sequence (MLS) probe signal from the left earpiece
receiver and records both the MLS reference and the earpiece microphone
response to a 2-channel WAV file, for impulse-response measurement (e.g. ear
canal / middle ear probing).

## Hardware

- Tympan RevE + EarpieceShield
- Sample rate: 48000 Hz, block size: 128

## Output

2-channel WAV file on the SD card:
- Channel 0 (left): averaged left-earpiece PDM mic signal
- Channel 1 (right): MLS reference signal sent to the speaker

## Usage

Open the Serial Monitor (115200 baud) and send `h` for the full command list.

Key commands:

| Command     | Action |
|-------------|--------|
| `p` / `P`   | Start / stop MLS playback |
| `a <val>`   | Set amplitude, `0.0`-`1.0` |
| `len <val>` | Set MLS length in samples (snaps to nearest valid value, 1023-65535) |
| `r` / `s`   | Start / stop SD recording |
| `v <val>`   | Set output volume (dB) |
| `i` / `I`   | Increase / decrease input gain |
| `g`         | Print current settings |

## Analysis

The `analysis/` folder has Python scripts for post-processing recordings:

- `mls_impulse.py` — deconvolves a recording into impulse responses. Requires
  `--mls_length` to match whatever length the firmware used for that recording.
  Pass `--save_ir <file.npz>` to save the measurement (fs, mls_length, mean_ir,
  all_irs, peak_idx) for later PCA training; add `--no_plot` for batch/headless
  processing of many recordings.
- `train_pca_feedback_model.py` — fits a PCA model across multiple `--save_ir`
  measurements (ideally from physically distinct conditions: insertion depth,
  vent size, subject/coupler, environment) and writes a `pca_model.csv` model
  file. See `AudioPassThru_FeedbackPEMAFC_ConstrainedPCA` (sibling Arduino project)
  for the constrained-adaptation AFC that loads this file from the SD card and
  restricts the adaptive feedback-path estimate to that PCA subspace.
- `split_channels.py` — splits a 2-channel WAV into separate mic/reference mono files.

Both `mls_impulse.py` and `train_pca_feedback_model.py` need `numpy` and
`matplotlib` (unless run with `--no_plot`); `mls_impulse.py` also optionally
uses `scipy` for `--highpass` filtering.

Typical workflow to build a constrained-AFC model:

```
# 1. Record one MLS measurement per physical condition with MLS_Probe_Tympan
#    (reposition earpiece / change vent / etc. between recordings).

# 2. Extract + save each measurement's impulse response:
python mls_impulse.py rec1.wav --mls_length 1023 --save_ir rec1.npz --no_plot
python mls_impulse.py rec2.wav --mls_length 1023 --save_ir rec2.npz --no_plot
...

# 3. Fit the PCA model across all measurements:
python train_pca_feedback_model.py rec*.npz --afl 256 --out pca_model.csv

# 4. Copy pca_model.csv to the Tympan's SD card and load it in
#    AudioPassThru_FeedbackPEMAFC_ConstrainedPCA ('load pca_model.csv', or set as
#    the sketch's default).
```

## License

MIT. Use at your own risk.
