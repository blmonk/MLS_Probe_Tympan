# MLS_Probe_Tympan

Arduino sketch for the [Tympan](https://www.tympan.org/) RevE + EarpieceShield
that plays a Maximum Length Sequence (MLS) probe signal from the left earpiece
receiver and records both the MLS reference and the earpiece microphone
response to a 2-channel WAV file, for impulse-response measurement (e.g. ear
canal / middle ear probing).

## Hardware

- Tympan RevE + EarpieceShield
- Sample rate: 44100 Hz, block size: 128

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
- `split_channels.py` — splits a 2-channel WAV into separate mic/reference mono files.

Both require `numpy`; `mls_impulse.py` also needs `matplotlib` and (optionally) `scipy`.

## License

MIT. Use at your own risk.
