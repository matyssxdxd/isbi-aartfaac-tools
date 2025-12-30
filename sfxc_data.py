import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from astropy.time import Time
# from vlsr import lsr

from vex import Vex
from sfxcdata import SFXCData

os.environ['TZ'] = 'UTC'
time.tzset()



def dopler(observed_frequency, velocity_receiver, base_frequency):
    """

    :param observed_frequency: observed frequency
    :param velocity_receiver: velocity of receiver
    :param base_frequency: laboratory determined frequency
    :return: velocity of source
    """
    speed_of_light = scipy.constants.speed_of_light
    velocity_source = (-((observed_frequency / base_frequency) - 1) *
                       speed_of_light + (velocity_receiver * 1000)) / 1000
    return velocity_source


def get_pols(ch):
    pol1 = ch[2]
    pol2 = ch[3]
    return (pol1, pol2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot cross-correlations from SFXC .cor file")
    parser.add_argument("cor_file", nargs='?', default="./B023.cor_0002", help="Path to .cor file")
    parser.add_argument("--baseline", nargs=2, metavar=("STA1", "STA2"), default=['Ib', 'Ir'],
                        help="Baseline to plot (two station names)")
    parser.add_argument("--channels", nargs='+', type=int, default=None,
                        help="List of channel freqnr indices to plot individually (e.g. 4 5)")
    args = parser.parse_args()

    cor_file = args.cor_file
    data = SFXCData(cor_file)

    # Print full file info summary
    def print_sfxc_info(d):
        print("===== SFXC .cor file info =====")
        try:
            print(f"input file: {d.inputfile.name}")
        except Exception:
            pass
        gh = getattr(d, 'global_header', None)
        if gh is not None:
            print("-- global header --")
            for name in gh._fields:
                print(f"{name}: {getattr(gh, name)}")
        print(f"nchan: {getattr(d, 'nchan', None)}")
        print(f"integration_time: {getattr(d, 'integration_time', None)}")
        print(f"start_time: {getattr(d, 'start_time', None)}")
        print(f"declared stations (exp_stations): {getattr(d, 'exp_stations', None)}")
        print(f"declared sources (exp_sources): {getattr(d, 'exp_sources', None)}")
        print(f"stations in current integration: {getattr(d, 'stations', None)}")
        print(f"channels (unique channel definitions): {len(getattr(d, 'channels', []))}")
        baselines = list(getattr(d, 'vis', {}).keys())
        print(f"number of baselines: {len(baselines)}")
        print(f"baselines (sample up to 10): {baselines[:10]}")
        print(f"current_slice: {getattr(d, 'current_slice', None)}")
        # show a small sample of visibility shapes and weights
        print("-- sample visibilities (baseline -> channel -> (shape, weight)) --")
        for bl in baselines[:5]:
            try:
                chmap = d.vis[bl]
            except Exception:
                continue
            sample = {}
            for i, (ch, visobj) in enumerate(chmap.items()):
                if i >= 5:
                    break
                sample[str(ch)] = (getattr(visobj.vis, 'shape', None), getattr(visobj, 'weight', None))
            print(f"{bl}: {sample}")
        print("===== end file info =====")

    print_sfxc_info(data)

    # choose baseline key tuple in the same order as the file uses
    target_bl = tuple(args.baseline)
    baselines = list(data.vis.keys())
    if target_bl not in baselines:
        # try reversed order
        target_bl = (args.baseline[1], args.baseline[0])
        if target_bl not in baselines:
            raise KeyError(f"Baseline {args.baseline} not found in data. Available: {baselines}")

    # accumulate integrations for the chosen baseline and its autocorrelations
    baseline_keys = list(data.vis.keys())
    baselines_to_acc = [target_bl]
    # autocorrelations for each station in the baseline
    ib_bl = (target_bl[0], target_bl[0])
    ir_bl = (target_bl[1], target_bl[1])
    if ib_bl in baseline_keys and ib_bl not in baselines_to_acc:
        baselines_to_acc.append(ib_bl)
    if ir_bl in baseline_keys and ir_bl not in baselines_to_acc:
        baselines_to_acc.append(ir_bl)

    baseline_integr = {bl: {} for bl in baselines_to_acc}

    # first integration already loaded in SFXCData.__init__
    while True:
        for bl in baselines_to_acc:
            vis_dict = data.vis.get(bl, {})
            for ch, visobj in vis_dict.items():
                try:
                    if ch not in baseline_integr[bl]:
                        baseline_integr[bl][ch] = visobj.vis.copy()
                    else:
                        baseline_integr[bl][ch] += visobj.vis
                except Exception:
                    baseline_integr[bl][ch] = visobj.vis.copy()

        if not data.next_integration():
            break

    # Diagnostics: list available freqnr values and CrossChannel keys for target baseline
    from collections import Counter, defaultdict
    freq_counter = Counter()
    freq_map = defaultdict(list)
    for ch in baseline_integr[target_bl].keys():
        freq_counter[ch.freqnr] += 1
        freq_map[ch.freqnr].append(ch)

    print("Available freqnr values and counts (for target baseline):")
    for fn in sorted(freq_counter.keys()):
        print(f"  freqnr={fn}: count={freq_counter[fn]}, entries={freq_map[fn]}")

    # If channels specified, plot each requested freqnr separately (or absolute spectral bins)
    pol_map = {0: 'RR', 1: 'LL'}
    if args.channels:
        req = list(args.channels)
        freq_keys = sorted(freq_counter.keys())
        if all(r in freq_keys for r in req):
            mode = 'freqnr'
        elif all(isinstance(r, int) and 0 <= r < data.nchan for r in req):
            mode = 'spectral'
        else:
            print(f"Requested channels {req} do not match freqnr keys {freq_keys} or spectral range 0..{data.nchan-1}")
            return

        if mode == 'freqnr':
            sel_freqnrs = set(req)
            for freqnr in sel_freqnrs:
                for polnum, polname in pol_map.items():
                    # gather cross-spectrum for this freqnr and polarization from target baseline
                    cross_list = [arr for ch, arr in baseline_integr[target_bl].items() if ch.freqnr == freqnr and ch.pol1 == ch.pol2 == polnum]
                    ib_list = [arr for ch, arr in baseline_integr.get(ib_bl, {}).items() if ch.freqnr == freqnr and ch.pol1 == ch.pol2 == polnum]
                    ir_list = [arr for ch, arr in baseline_integr.get(ir_bl, {}).items() if ch.freqnr == freqnr and ch.pol1 == ch.pol2 == polnum]

                    if not cross_list:
                        print(f"No data for freqnr={freqnr} pol={polname} on baseline {target_bl}")
                        continue

                    vis_cross = np.sum(cross_list, axis=0)
                    vis_ib = np.sum(ib_list, axis=0) if ib_list else None
                    vis_ir = np.sum(ir_list, axis=0) if ir_list else None

                    n = len(vis_cross)
                    lags = np.fft.fftshift(np.fft.ifft(vis_cross))
                    lag_amp = np.abs(lags)
                    channels_axis = np.arange(n)

                    # create 4-row plot: Ib auto, Ir auto, cross lag amp, cross phase
                    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
                    fig.suptitle(f"Baseline {target_bl} freqnr={freqnr} pol={polname}")

                    axs[0].set_title('Ib auto amplitude')
                    if vis_ib is not None:
                        axs[0].plot(channels_axis, np.abs(vis_ib))
                    else:
                        axs[0].text(0.5, 0.5, 'No Ib auto', ha='center')

                    axs[1].set_title('Ir auto amplitude')
                    if vis_ir is not None:
                        axs[1].plot(channels_axis, np.abs(vis_ir))
                    else:
                        axs[1].text(0.5, 0.5, 'No Ir auto', ha='center')

                    axs[2].set_title('Lag amplitude (IFFT of cross-spectrum)')
                    t = np.arange(-n // 2, n // 2)
                    axs[2].plot(t, lag_amp)

                    axs[3].set_title('Phase per channel (deg) [cross]')
                    axs[3].scatter(channels_axis, np.angle(vis_cross, deg=True))

                    plt.tight_layout()
                    plt.show()
        else:
            # spectral mode: plot absolute spectral bins across contributing CrossChannels
            spectral_indices = req
            for idx in spectral_indices:
                for polnum, polname in pol_map.items():
                    vals_cross = [arr[idx] for ch, arr in baseline_integr[target_bl].items() if ch.pol1 == ch.pol2 == polnum and 0 <= idx < arr.size]
                    vals_ib = [arr[idx] for ch, arr in baseline_integr.get(ib_bl, {}).items() if ch.pol1 == ch.pol2 == polnum and 0 <= idx < arr.size]
                    vals_ir = [arr[idx] for ch, arr in baseline_integr.get(ir_bl, {}).items() if ch.pol1 == ch.pol2 == polnum and 0 <= idx < arr.size]

                    if not vals_cross:
                        print(f"No spectral data at index={idx} for pol={polname} on baseline {target_bl}")
                        continue

                    vals_cross = np.array(vals_cross)
                    vals_ib = np.array(vals_ib) if vals_ib else None
                    vals_ir = np.array(vals_ir) if vals_ir else None

                    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
                    fig.suptitle(f"Baseline {target_bl} spectral_index={idx} pol={polname}")

                    axs[0].set_title('Ib auto: amplitude across contributing channels')
                    if vals_ib is not None:
                        axs[0].plot(np.abs(vals_ib))
                    else:
                        axs[0].text(0.5, 0.5, 'No Ib auto', ha='center')

                    axs[1].set_title('Ir auto: amplitude across contributing channels')
                    if vals_ir is not None:
                        axs[1].plot(np.abs(vals_ir))
                    else:
                        axs[1].text(0.5, 0.5, 'No Ir auto', ha='center')

                    axs[2].set_title('Cross amplitude across contributing channels')
                    axs[2].plot(np.abs(vals_cross))

                    axs[3].set_title('Cross phase (deg) across contributing channels')
                    axs[3].scatetr(np.arange(len(vals_cross)), np.angle(vals_cross, deg=True))

                    plt.tight_layout()
                    plt.show()
    else:
        # Sum cross-spectra per polarization (RR, LL) using accumulated target baseline
        summed = {p: None for p in pol_map.values()}
        for ch, arr in baseline_integr[target_bl].items():
            if ch.pol1 == ch.pol2 and ch.pol1 in pol_map:
                p = pol_map[ch.pol1]
                if summed[p] is None:
                    summed[p] = arr.copy()
                else:
                    summed[p] += arr

        # also sum autocorrelations if available
        summed_ib = {p: None for p in pol_map.values()} if ib_bl in baseline_integr else None
        summed_ir = {p: None for p in pol_map.values()} if ir_bl in baseline_integr else None
        if summed_ib is not None:
            for ch, arr in baseline_integr[ib_bl].items():
                if ch.pol1 == ch.pol2 and ch.pol1 in pol_map:
                    p = pol_map[ch.pol1]
                    if summed_ib[p] is None:
                        summed_ib[p] = arr.copy()
                    else:
                        summed_ib[p] += arr
        if summed_ir is not None:
            for ch, arr in baseline_integr[ir_bl].items():
                if ch.pol1 == ch.pol2 and ch.pol1 in pol_map:
                    p = pol_map[ch.pol1]
                    if summed_ir[p] is None:
                        summed_ir[p] = arr.copy()
                    else:
                        summed_ir[p] += arr

        # Plot summed results including autos
        for polnum, pol in pol_map.items():
            vis = summed.get(pol)
            vis_ib = summed_ib.get(pol) if summed_ib is not None else None
            vis_ir = summed_ir.get(pol) if summed_ir is not None else None
            if vis is None:
                print(f"No {pol} data for baseline {target_bl}")
                continue

            n = len(vis)
            lags = np.fft.fftshift(np.fft.ifft(vis))
            lag_amp = np.abs(lags)
            channels_axis = np.arange(n)

            fig, axs = plt.subplots(4, 1, figsize=(10, 12))
            fig.suptitle(f"Baseline {target_bl} pol={pol}")

            axs[0].set_title('Ib auto amplitude')
            if vis_ib is not None:
                axs[0].plot(channels_axis, np.abs(vis_ib))
            else:
                axs[0].text(0.5, 0.5, 'No Ib auto', ha='center')

            axs[1].set_title('Ir auto amplitude')
            if vis_ir is not None:
                axs[1].plot(channels_axis, np.abs(vis_ir))
            else:
                axs[1].text(0.5, 0.5, 'No Ir auto', ha='center')

            axs[2].set_title('Lag amplitude (IFFT of cross-spectrum)')
            t = np.arange(-n // 2, n // 2)
            axs[2].plot(t, lag_amp)

            axs[3].set_title('Phase per channel (deg)')
            axs[3].scatter(channels_axis, np.angle(vis, deg=True))

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
    sys.exit()