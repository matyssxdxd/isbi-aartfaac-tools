
import os
import glob
import numpy as np

from utils.process_data import read_visibility_file

BL_CROSS = 1
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
    path = "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/"
    files = sorted(glob.glob(os.path.join(path, "*.out")))
    guard = 3

    for file in files:
        headers, visibilities = read_visibility_file(file, normalize=True)

        cross_stack = np.asarray([v[BL_CROSS] for v in visibilities], dtype=np.complex64)
        w = np.asarray([h.weights[BL_CROSS] for h in headers], dtype=np.float64)

        cross_avg = np.tensordot(w, cross_stack, axes=(0, 0)) / w.sum()

        chan_bw_hz = headers[0].channel_bandwidth

        print(f"\n{os.path.basename(file)} vis_weight_sum={w.sum():.0f} vis_weight_mean={w.mean():.1f}")

        for p, pol in enumerate(POLS):
            spec = np.conj(cross_avg[:, p])
            amp, snr, lag_bins, lag_spec = fringe_metrics(spec, chan_bw_hz)
            print(f"{pol}: amp={amp:.4e}, snr={snr:.2f}, lag={lag_bins:+d}, bins({lag_spec*1e9:+.2f} ns)")








