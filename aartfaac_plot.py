#!/usr/bin/env python3

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from sfxcdata import SFXCData
from utils.process_data import read_visibility_file

POLS = ["RR", "RL", "LR", "LL"]
POL_MAP = {(0, 0): "RR", (0, 1): "RL", (1, 0): "LR", (1, 1): "LL"}


def _normalize_paths(path_or_paths):
    if isinstance(path_or_paths, str):
        return [path_or_paths]
    return list(path_or_paths)


def _collect_sfxc_files(path_or_paths):
    files = []
    for path in _normalize_paths(path_or_paths):
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.cor*"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            files.extend(sorted(glob.glob(path)))
    # Keep order stable and deduplicate.
    return list(dict.fromkeys(files))


def _collect_out_files(path_or_paths):
    files = []
    for path in _normalize_paths(path_or_paths):
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.out"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            files.extend(sorted(glob.glob(path)))

    def _sort_key(path):
        base = os.path.basename(path)
        digits = "".join(ch for ch in base if ch.isdigit())
        return (int(digits) if digits else 10**9, base)

    files = sorted(list(dict.fromkeys(files)), key=_sort_key)
    return files


def _select_subband_pairs(subbands):
    # Map subband index to a single (freqnr, sideband) combination.
    # 1 -> (0,0), 2 -> (0,1), 3 -> (1,0), 4 -> (1,1), ...
    selected_pairs = []
    seen_pairs = set()
    for sb in subbands:
        if sb < 1:
            raise ValueError(f"Invalid subband value: {sb}. Subbands must start at 1.")
        idx = sb - 1
        pair = (idx // 2, idx % 2)
        if pair not in seen_pairs:
            selected_pairs.append(pair)
            seen_pairs.add(pair)
    return selected_pairs, seen_pairs


def _load_sfxc_by_key(sfxc_corr_paths, subbands, baseline=("Ib", "Ir")):
    files = _collect_sfxc_files(sfxc_corr_paths)
    if not files:
        raise FileNotFoundError("No SFXC correlation files found")

    selected_pairs, seen_pairs = _select_subband_pairs(subbands)

    by_key = {}

    for file in files:
        sfxc = SFXCData(file)

        def collect_current():
            if baseline not in sfxc.vis:
                return
            for chan, vis in sfxc.vis[baseline].items():
                if (chan.freqnr, chan.sideband) not in seen_pairs:
                    continue
                pol = POL_MAP.get((chan.pol1, chan.pol2))
                if pol is None:
                    continue
                key = (chan.freqnr, chan.sideband, pol)
                by_key.setdefault(key, []).append(np.asarray(vis.vis, dtype=np.complex64))

        collect_current()
        while sfxc.next_integration():
            collect_current()

    if not by_key:
        raise ValueError(
            f"No SFXC visibilities found for baseline {baseline} and subbands {subbands}"
        )

    return selected_pairs, by_key


def _load_my_visibility_blocks(my_data_paths):
    files = _collect_out_files(my_data_paths)
    if not files:
        raise FileNotFoundError("No .out files found for your data")

    vis_blocks = []
    for file in files:
        _, visibilities = read_visibility_file(file, normalize=True)
        if not visibilities:
            continue
        vis = np.asarray(visibilities, dtype=np.complex64)
        vis_blocks.append((file, vis))

    if not vis_blocks:
        raise ValueError("No integrations found in .out files")

    return vis_blocks


def _build_integration_windows(total_integrations, integration=None, integration_batch_size=None):
    if total_integrations <= 0:
        raise ValueError("No integrations available to plot")

    if integration is not None and integration_batch_size is not None:
        raise ValueError("Set either integration or integration_batch_size, not both")

    if integration is not None:
        if integration < 0 or integration >= total_integrations:
            raise ValueError(
                f"Integration {integration} out of range. Available: 0-{total_integrations - 1}"
            )
        return [(integration, integration + 1, f"Integration {integration}")]

    if integration_batch_size is not None:
        if integration_batch_size <= 0:
            raise ValueError("integration_batch_size must be a positive integer")

        windows = []
        for start in range(0, total_integrations, integration_batch_size):
            end = min(start + integration_batch_size, total_integrations)
            windows.append((start, end, f"Integrations {start}-{end - 1}"))
        return windows

    return [(0, total_integrations, "Averaged")]


def _build_sfxc_pol_vectors_for_window(selected_pairs, by_key, window_start, window_end):
    pol_vectors = {}
    for pol in POLS:
        chunks = []
        for freqnr, sideband in selected_pairs:
            key = (freqnr, sideband, pol)
            integrations = by_key.get(key)
            if not integrations:
                continue

            window = integrations[window_start:window_end]
            if not window:
                continue
            vec = np.mean(window, axis=0)

            # Hardcoded SFXC rule: odd subbands -> flip, even subbands -> conjugate.
            subband_number = 2 * freqnr + sideband + 1
            if subband_number % 2 == 1:
                vec = np.flip(vec)
            else:
                vec = np.conj(vec)

            vec = vec[1:-1]

            chunks.append(np.asarray(vec, dtype=np.complex64))

        if chunks:
            pol_vectors[pol] = np.concatenate(chunks)
        else:
            pol_vectors[pol] = np.array([], dtype=np.complex64)

    return pol_vectors


def _build_my_pol_vectors_for_window(vis_blocks, window_start, window_end):
    pol_chunks = {pol: [] for pol in POLS}

    for _, vis in vis_blocks:
        selected = np.mean(vis[window_start:window_end], axis=0)

        # Baseline index 1 is the cross-correlation baseline for 2 stations.
        cross = selected[1]
        pol_chunks["RR"].append(np.asarray(cross[:, 0]).conj())
        pol_chunks["RL"].append(np.asarray(cross[:, 1]).conj())
        pol_chunks["LR"].append(np.asarray(cross[:, 2]).conj())
        pol_chunks["LL"].append(np.asarray(cross[:, 3]).conj())

    pol_vectors = {}
    for pol in POLS:
        if pol_chunks[pol]:
            pol_vectors[pol] = np.concatenate(pol_chunks[pol])
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

        ax_phase.scatter(x, phase, label=pol, s=5)
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

def _plot_phase_diff(ax_phase, pol_vectors_sfxc, pol_vectors_mine, title):
    for pol in POLS:
        data_sfxc = pol_vectors_sfxc[pol]
        data_mine = pol_vectors_mine[pol]
        n = min(data_sfxc.size, data_mine.size)
        if n == 0:
            continue

        x = np.arange(n)
        phase = np.angle(data_sfxc[:n] * np.conj(data_mine[:n]), deg=True)
        ax_phase.scatter(x, phase, label=pol, s=5)


    ax_phase.set_title(title)
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_ylim(-200, 200)
    ax_phase.set_xlabel("Channel index")
    ax_phase.legend()

def _save_pol_vectors_to_text(pol_vectors, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for pol in POLS:
            file.write(f"{pol}:\n")
            for sample in pol_vectors[pol]:
                file.write(f"{sample.real:.8e} {sample.imag:.8e}\n")
            file.write("\n")


def plot_sfxc_vs_mine(
    sfxc_corr_paths,
    my_data_paths,
    title,
    sfxc_subbands,
    baseline=("Ib", "Ir"),
    integration=None,
    integration_batch_size=None,
    sfxc_text_output="sfxc_data_E011_No0017.txt",
    my_text_output="aartfaac_data_E011_No0017.txt",
):
    """Plot SFXC data (left) and your data (right) side by side.

    Args:
        sfxc_corr_paths: SFXC .cor file path, directory, glob, or list of these.
        my_data_paths: Your .out file path, directory, glob, or list of these.
        title: Figure title.
        sfxc_subbands: Iterable of subbands to include from SFXC.
            Example: [1, 2, 3, 4, 5, 6, 7, 8] -> includes all freqnr/sideband
            combinations found for those subbands in one combined plot.
            Fixed mapping is 1-indexed:
            1 -> (0,0), 2 -> (0,1), 3 -> (1,0), 4 -> (1,1), ...
        baseline: Baseline tuple in SFXC data.
        integration: If set, use this integration index. If None, average all.
        integration_batch_size: If set, plot consecutive batches of this size
            (e.g. 5 -> 0-4, 5-9, ...), averaging integrations within each batch.
        sfxc_text_output: Output text file for SFXC polarization data.
        my_text_output: Output text file for ISBI-AARTFAAC polarization data.
    """
    selected_pairs, sfxc_by_key = _load_sfxc_by_key(
        sfxc_corr_paths=sfxc_corr_paths,
        subbands=sfxc_subbands,
        baseline=baseline,
    )
    my_vis_blocks = _load_my_visibility_blocks(my_data_paths=my_data_paths)

    sfxc_counts = [len(integrations) for integrations in sfxc_by_key.values() if integrations]
    if not sfxc_counts:
        raise ValueError("No SFXC integrations found after filtering")
    sfxc_total_integrations = min(sfxc_counts)

    my_counts = [vis.shape[0] for _, vis in my_vis_blocks]
    if not my_counts:
        raise ValueError("No AARTFAAC integrations found after filtering")
    my_total_integrations = min(my_counts)

    total_integrations = min(sfxc_total_integrations, my_total_integrations)
    windows = _build_integration_windows(
        total_integrations=total_integrations,
        integration=integration,
        integration_batch_size=integration_batch_size,
    )

    for window_start, window_end, integration_label in windows:
        sfxc_pol_vectors = _build_sfxc_pol_vectors_for_window(
            selected_pairs=selected_pairs,
            by_key=sfxc_by_key,
            window_start=window_start,
            window_end=window_end,
        )
        my_pol_vectors = _build_my_pol_vectors_for_window(
            vis_blocks=my_vis_blocks,
            window_start=window_start,
            window_end=window_end,
        )
        # _save_pol_vectors_to_text(sfxc_pol_vectors, sfxc_text_output)
        # _save_pol_vectors_to_text(my_pol_vectors, my_text_output)

        # fig_phase, ax_phase = plt.subplots(1, 1, figsize=(16, 10))
        # _plot_phase_diff(ax_phase, sfxc_pol_vectors, my_pol_vectors, "SFXC vs AARTFAAC phase")
        # fig_phase.suptitle(f"{title} | {integration_label}")
        # fig_phase.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig, axs = plt.subplots(3, 2, figsize=(16, 10))
        _plot_dataset(axs[0, 0], axs[1, 0], axs[2, 0], sfxc_pol_vectors, "SFXC")
        _plot_dataset(axs[0, 1], axs[1, 1], axs[2, 1], my_pol_vectors, "ISBI-AARTFAAC")
        fig.suptitle(f"{title} | {integration_label}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    plot_sfxc_vs_mine(
        sfxc_corr_paths="./E011/E011.cor_0001",
        my_data_paths="./results/E011_No0002/63_TEST/",
        title="SFXC vs AARTFAAC | E011 No0002 | 2 sec integration",
        sfxc_subbands=[1, 2, 3, 4, 5, 6, 7, 8],
#        integration_batch_size=10
    )
