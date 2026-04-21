#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.sfxcdata import SFXCData

POLS = ["RR", "RL", "LR", "LL"]
POL_MAP = {(0, 0): "RR", (0, 1): "RL", (1, 0): "LR", (1, 1): "LL"}


def _choose_baseline(visibilities):
    cross_baselines = sorted(bl for bl in visibilities if bl[0] != bl[1])
    if cross_baselines:
        return cross_baselines[0]

    return sorted(visibilities)[0] if visibilities else None


def _read_sfxc_pol_vectors(cor_file):
    if not os.path.isfile(cor_file):
        raise FileNotFoundError(f"SFXC correlation file not found: {cor_file}")

    by_key = {}
    available_baselines = set()
    available_pairs = set()
    sfxc = SFXCData(cor_file)

    def collect_current_integration():
        available_baselines.update(sfxc.vis.keys())
        baseline = _choose_baseline(sfxc.vis)
        if baseline is None:
            return

        for chan, vis in sfxc.vis[baseline].items():
            pol = POL_MAP.get((chan.pol1, chan.pol2))
            if pol is None:
                continue

            pair = (chan.freqnr, chan.sideband)
            available_pairs.add(pair)
            by_key.setdefault((*pair, pol), []).append(
                np.asarray(vis.vis, dtype=np.complex64)
            )

    collect_current_integration()
    while sfxc.next_integration():
        collect_current_integration()

    if not by_key:
        raise ValueError(
            "No SFXC visibilities found. "
            f"Available baselines: {sorted(available_baselines)}"
        )

    selected_pairs = sorted(available_pairs)
    pol_vectors = {}
    for pol in POLS:
        chunks = []
        for freqnr, sideband in selected_pairs:
            integrations = by_key.get((freqnr, sideband, pol))
            if not integrations:
                continue

            vec = np.mean(integrations, axis=0)

            # Match aartfaac_plot.py: odd subbands are flipped, even subbands
            # are conjugated before plotting.
            subband_number = 2 * freqnr + sideband + 1
            if subband_number % 2 == 1:
                vec = np.flip(vec)
            else:
                vec = np.conj(vec)

            chunks.append(np.asarray(vec, dtype=np.complex64))

        if chunks:
            pol_vectors[pol] = np.concatenate(chunks)
        else:
            pol_vectors[pol] = np.array([], dtype=np.complex64)

    return pol_vectors


def _plot_dataset(ax_phase, ax_amp, ax_lag, pol_vectors, title):
    for pol in POLS:
        data = pol_vectors[pol]
        if data.size == 0:
            continue

        x = np.arange(data.size)
        phase = np.angle(data, deg=True)
        ampl = np.abs(data)

        ax_phase.scatter(x, phase, label=pol, s=3)
        ax_amp.plot(x, ampl, label=pol, linewidth=1)

        corr = np.fft.irfft(data)
        corr = np.fft.fftshift(corr)
        lags = np.arange(-len(corr) // 2, len(corr) // 2)[: len(corr)]
        ax_lag.plot(lags, np.abs(corr), label=pol, linewidth=1)

    ax_phase.set_title(title)
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_ylim(-200, 200)
    ax_phase.set_xlabel("Channel index")
    ax_phase.legend()

    ax_amp.set_ylabel("Amplitude")
    ax_amp.set_xlabel("Channel index")
    ax_amp.legend()

    ax_lag.set_ylabel("Lag amplitude")
    ax_lag.set_xlabel("Lag")
    ax_lag.legend()


def plot_sfxc(cor_file):
    pol_vectors = _read_sfxc_pol_vectors(cor_file)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    _plot_dataset(axs[0], axs[1], axs[2], pol_vectors, "SFXC")
    fig.suptitle(f"SFXC | {os.path.basename(cor_file)} | Averaged")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot all polarization data from one SFXC .cor file."
    )
    parser.add_argument("cor_file", help="Path to the SFXC .cor file")
    args = parser.parse_args()

    plot_sfxc(args.cor_file)


if __name__ == "__main__":
    main()
