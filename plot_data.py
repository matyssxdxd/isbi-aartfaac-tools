from utils.read_data import read_all, baseline_index
from utils.process_data import extract_weights, normalize_cross, normalize_auto, weighted_average_integrations
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ISBI-AARTFAAC correlator data from an output directory."
    )
    parser.add_argument(
        "output_path",
        help="Directory containing correlator .out files.",
    )
    parser.add_argument(
        "--stations", default=None,
        help="Comma-separated station names in the SAME order as ISBI's -i, e.g. Ef,Hh,Ir. "
             "The .out header carries no names. Default: 0, 1, 2, ...",
    )
    parser.add_argument(
        "--save", default=None,
        help="Directory to write PNGs to instead of showing the figures.",
    )
    args = parser.parse_args()

    output_path = args.output_path
    output_files = sorted(glob.glob(os.path.join(output_path, "*.out")))

    headers, visibilities = read_all(output_files)

    # Sky-frequency order, NOT filename order: sorted() puts subband_10 before
    # subband_2, which would scramble the stitched frequency axis.
    order = np.argsort([h[0].first_channel_frequency for h in headers])

    processed_data = []
    for k in order:
        h, vis = headers[k], visibilities[k]
        w = extract_weights(h)
        # norm = normalize_cross(vis)
        # norm, _ = normalize_auto(norm)
        averaged, _ = weighted_average_integrations(vis, w)
        processed_data.append(averaged)

    processed_data = np.array(processed_data)
    nsub, nchan, nbl = processed_data.shape[:3]
    nst = headers[0][0].nr_receivers
    processed_data = processed_data.reshape(nsub * nchan, nbl, 2, 2)

    if args.stations:
        names = [s.strip() for s in args.stations.split(",")]
        if len(names) != nst:
            raise SystemExit(f"--stations has {len(names)} names but the data has {nst} stations")
    else:
        names = [str(s) for s in range(nst)]
    print(f"stations {', '.join(names)}, subbands {nsub}, channels {nsub * nchan}")

    for s1 in range(nst):
        for s2 in range(s1 + 1, nst):
            bl = baseline_index(s1, s2)
            data = {
                    "XX": np.conj(processed_data[:, bl, 0, 0]),
                    "XY": np.conj(processed_data[:, bl, 0, 1]),
                    "YX": np.conj(processed_data[:, bl, 1, 0]),
                    "YY": np.conj(processed_data[:, bl, 1, 1])
            }

            plt.figure(figsize=(12, 8))

            for pol in data:
                amp = np.abs(data[pol])
                phase = np.angle(data[pol], deg=True)
                x = np.arange(len(amp))
                corr = np.fft.irfft(data[pol])
                corr = np.fft.fftshift(corr)
                lags = np.arange(-len(corr) // 2, len(corr) // 2)[: len(corr)]

                plt.subplot(311)
                plt.title(f"Amplitude  (baseline {names[s1]}-{names[s2]})")
                plt.plot(x, amp, label=pol, linewidth=1)

                plt.subplot(312)
                plt.title("Phase")
                plt.scatter(x, phase, label=pol, s=5)

                plt.subplot(313)
                plt.title("Lag")
                plt.plot(lags, np.abs(corr), label=pol, linewidth=1)

            for panel in (311, 312, 313):
                plt.subplot(panel)
                plt.legend(fontsize=8)
            plt.tight_layout()
            if args.save:
                os.makedirs(args.save, exist_ok=True)
                plt.savefig(os.path.join(args.save, f"cross_{names[s1]}-{names[s2]}.png"), dpi=150)

    for s in range(nst):
        bl = baseline_index(s, s)
        auto_data = {
                "XX": processed_data[:, bl, 0, 0],
                "YY": processed_data[:, bl, 1, 1],
        }

        plt.figure(figsize=(12, 8))

        for pol in auto_data:
            amp = np.abs(auto_data[pol])
            x = np.arange(len(amp))

            plt.title(f"Amplitude  (autocorrelation, {names[s]})")
            plt.plot(x, amp, label=pol, linewidth=1)

        plt.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(os.path.join(args.save, f"auto_{names[s]}.png"), dpi=150)

    if not args.save:
        plt.show()
