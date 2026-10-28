#!/usr/bin/env python3

import os
import glob
import numpy as np

from utils.helpers import parse_arguments
from utils.read_data import read_subband, baseline_index

POLS = ["RR", "RL", "LR", "LL"]
POL_MAP = {(0, 0): "RR", (0, 1): "RL", (1, 0): "LR", (1, 1): "LL"}

def fringe_metrics(spec, chan_bw_hz, guard=3):
    lag = np.fft.fftshift(np.fft.ifft(spec))
    lag_abs = np.abs(lag)

    peak_idx = int(np.argmax(lag_abs))
    fringe_amp = float(lag_abs[peak_idx])

    lag_bins = peak_idx - (lag_abs.size // 2)
    lag_sec = lag_bins / (spec.size * abs(chan_bw_hz))

    mask = np.ones(lag_abs.size, dtype=bool)
    lo = max(0, peak_idx - guard)
    hi = min(lag_abs.size, peak_idx + guard + 1)
    mask[lo:hi] = False

    noise = lag[mask]
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2))) if noise.size else 0.0
    snr = fringe_amp / noise_rms if noise_rms > 0 else np.inf

    return fringe_amp, snr, lag_bins, lag_sec

if __name__ == "__main__":
    description = 'Print fringe metrics about ISBI-AARTFAAC .out files'
    arguments = {
        'out_path': 'Path to ISBI-AARTFAAC .out files'
    }

    args = parse_arguments(description, arguments)

    files = sorted(glob.glob(os.path.join(args.out_path, "*.out")))
    guard = 3

    for file in files:
        headers, visibilities = read_subband(file)
        nst = headers[0].nr_receivers
        chan_bw_hz = headers[0].channel_bandwidth
        print(f"\n{os.path.basename(file)}  ({nst} stations)")

        for s1 in range(nst):
            for s2 in range(s1 + 1, nst):
                bl = baseline_index(s1, s2)
                cross_stack = np.asarray(
                    [v[:, bl, :, :] for v in visibilities], dtype=np.complex64
                )
                # Clip: a negative weight is a baseline with no valid samples,
                # and --referenceStation writes 0 for baselines it skipped.
                w = np.clip(np.asarray([h.weights[bl] for h in headers], dtype=np.float64), 0, None)

                print(f"  baseline {s1}-{s2}: vis_weight_sum={w.sum():.0f} vis_weight_mean={w.mean():.1f}")
                if w.sum() == 0:
                    print("    no weight (fully flagged, or not produced under --referenceStation)")
                    continue

                cross_avg = np.tensordot(w, cross_stack, axes=(0, 0)) / w.sum()

                for polx in range(2):
                    for poly in range(2):
                        pol = POL_MAP[(polx, poly)]
                        spec = np.conj(cross_avg[:, polx, poly])
                        amp, snr, lag_bins, lag_spec = fringe_metrics(spec, chan_bw_hz)
                        print(f"    {pol}: amp={amp:.4e}, snr={snr:.2f}, lag={lag_bins:+d}, bins({lag_spec*1e9:+.2f} ns)")






