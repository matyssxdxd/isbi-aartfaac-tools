import argparse
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils.process_data import read_visibility_file
import sys

BL2IDX = {0: 'BL0', 1: 'BL0BL1', 2: 'BL1'}
IDX2BL = {'BL0': 0, 'BL0BL1': 1, 'BL1': 2}
    
POL2IDX = {0: 'RR', 1: 'RL', 2: 'LR', 3: 'LL'}
IDX2POL = {'RR': 0, 'RL': 1, 'LR': 2, 'LL': 3}

BASELINES = ['BL0', 'BL0BL1', 'BL1']
POLARIZATIONS = ['RR', 'RL', 'LR', 'LL']

WEIGHTS = []
# Bandwidth (Hz) used to convert lag bins to seconds
BANDWIDTH_HZ = 16e6
# Number of lag bins the user expects / wants to use for delay calculation
# (user requested n = 127)
USER_LAG_BINS = 255

def read_raw_data(file_paths):
    raw_data = []
    for file in file_paths:
        header, visibilities = read_visibility_file(file)
        for i, h in enumerate(header):
            weights = [w for w in h.weights if w != 0]
            WEIGHTS.append(weights)
        raw_data.append(visibilities)
    return np.array(raw_data)

def raw_data_to_dict(raw_data):
    n_subbands = raw_data.shape[0]
    n_vis = raw_data.shape[1]
    n_baselines= raw_data.shape[2]
    n_channels= raw_data.shape[3]
    n_polarizations= raw_data.shape[4]
    
    visibilities = {
        subband: {
            'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
            'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
            'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
        }
        for subband in range(n_subbands)
    }
    
    for subband in range(n_subbands):
        for vis in range(n_vis):
            for baseline in range(n_baselines):
                for pol in range(n_polarizations):
                    bkey = BL2IDX[baseline]
                    pkey = POL2IDX[pol]
                    visibilities[subband][bkey][pkey].append(
                        raw_data[subband][vis][baseline][:, pol]
                    )
                    
    return visibilities

def average_raw_data(raw_data):
    n_subbands = max(raw_data.keys()) + 1
    
    visibilities = {
        subband: {
            'BL0': {'RR': None, 'RL': None, 'LR': None, 'LL': None},
            'BL0BL1': {'RR': None, 'RL': None, 'LR': None, 'LL': None},
            'BL1': {'RR': None, 'RL': None, 'LR': None, 'LL': None}
        }
        for subband in range(n_subbands)
    }
    
    for subband in range(n_subbands):
        for baseline in BASELINES:
            blidx = IDX2BL[baseline]
            w = np.array(WEIGHTS)[:, blidx]
            for pol in POLARIZATIONS:
                data = np.array(raw_data[subband][baseline][pol], dtype=np.complex64)
                avg_data = np.average(data, axis=0, weights=w)
                visibilities[subband][baseline][pol] = avg_data
                
    return visibilities

def plot_data(data):
    n_subbands = max(data.keys()) + 1
    
    for subband in range(n_subbands):
        for pol in POLARIZATIONS:
            if pol in ['RR', 'LL']:
                fig, axs = plt.subplots(5, 1, figsize=(10, 15))
            else:
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                
            lags_spec = data[subband]['BL0BL1'][pol]
            corr = np.fft.irfft(lags_spec)
            corr = np.fft.fftshift(corr)
            corr_abs = np.abs(corr)

            n = len(corr_abs)
            # Symmetric lag axis for odd n
            t_lag = np.arange(-n//2, n//2 + 1)
            t_lag = t_lag[:n]  # in case n is even, trim to length n

            peak_idx = int(np.argmax(corr_abs))
            lag_bins = t_lag[peak_idx]
            delay_seconds = lag_bins / BANDWIDTH_HZ
            
            freq = 6675.69
            freq0 = 6675.69 - 16
            t = np.linspace(freq0, freq, 255)
            
            if pol in ['RR', 'LL']:
                axs[0].set_title('Amplitude/Frequency BL0 auto correlation')
                axs[0].plot(t, np.abs(data[subband]['BL0'][pol]), label='BL0')
                axs[0].set_xlabel("Frequency (MHz)")
                axs[0].set_ylabel("Amplitude")
                
                
                axs[1].set_title('Amplitude/Frequency BL1 auto correlation')
                axs[1].plot(t, np.abs(data[subband]['BL1'][pol]), label='BL0')
                axs[1].set_xlabel("Frequency (MHz)")
                axs[1].set_ylabel("Amplitude")
                
                
                axs[2].set_title('Ampltiude/Lag cross-correlation')
                axs[2].plot(t_lag, corr_abs)
                
                axs[3].set_title('Amplitude/Frequency cross-correlation')
                axs[3].plot(t, np.abs(data[subband]['BL0BL1'][pol]), label='BL0BL1')
                axs[3].set_xlabel("Frequency (MHz)")
                axs[3].set_ylabel("Amplitude")
                
                
                axs[4].set_title('Phase/Channel cross-correlation')
                axs[4].scatter(t, np.angle(data[subband]['BL0BL1'][pol], deg=True), label='BL0BL1')
                axs[4].set_xlabel("Frequency (MHz)")
                axs[4].set_ylabel("Phase (deg)")
            else:
                axs[0].set_title('Ampltiude/Lag cross-correlation')
                axs[0].plot(t_lag, corr_abs)
                
                axs[1].set_title('Amplitude/Frequency cross-correlation')
                axs[1].plot(t, np.abs(data[subband]['BL0BL1'][pol]), label='BL0BL1')
                axs[1].set_xlabel("Frequency (MHz)")
                axs[1].set_ylabel("Amplitude")
                
                axs[2].set_title('Phase/Frequency cross-correlation')
                axs[2].scatter(t, np.angle(data[subband]['BL0BL1'][pol], deg=True), label='BL0BL1')
                axs[2].set_xlabel("Frequency (MHz)")
                axs[2].set_ylabel("Phase (deg)")
                
            fig.suptitle(
                f'subband=3, pol={pol}, lag={lag_bins}, del={delay_seconds}'
            )
            
            print(f'pol={pol}, lag={lag_bins}, del={delay_seconds}')
            
            plt.tight_layout()
            plt.show()

def plot_data_single(data):
    n_subbands = max(data.keys()) + 1
    
    freq = 6675.69
    freq0 = 6675.69 - 16
    t = np.linspace(freq0, freq, 255)
    fig, axs = plt.subplots(2, 2, figsize=(10, 15))
    
    for subband in range(n_subbands):
        for baseline in BASELINES:
            for pol in POLARIZATIONS:
                if baseline == 'BL0' and pol in ['RR', 'LL']:
                    axs[0, 0].set_title('Amplitude/Channel ' + baseline + ' auto correlation')
                    axs[0, 0].plot(t, np.abs(data[subband][baseline][pol]), label=f'Pol={pol}')
                    axs[0, 0].legend()
                elif baseline == 'BL1' and pol in ['RR', 'LL']:
                    axs[0, 1].set_title('Amplitude/Channel ' + baseline + ' auto correlation')
                    axs[0, 1].plot(t, np.abs(data[subband][baseline][pol]), label=f'Pol={pol}')
                    axs[0, 1].legend()
                elif baseline == 'BL0BL1':
                    axs[1, 0].set_title('Amplitude/Channel cross-correlation')
                    axs[1, 0].plot(t, np.abs(data[subband][baseline][pol]), label=f'Pol={pol}')
                    axs[1, 0].legend()
                    
                    axs[1, 1].set_title('Phase/Channel cross-correlation')
                    axs[1, 1].scatter(t, np.angle(data[subband][baseline][pol], deg=True), label=f'Pol={pol}')
                    axs[1, 1].legend()
    plt.tight_layout()
    plt.show()                
    
def normalize_complex_array(a, eps=1e-12):
    """Normalize complex array so max amplitude is 1 (per array)."""
    amp = np.abs(a)
    max_amp = amp.max()
    if max_amp < eps:
        return a
    return a / max_amp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ctrl", help="Control file")

    args = parser.parse_args()

    with open(args.ctrl, 'r') as ctrl:
        ctrl_f = json.load(ctrl)

    out_files = glob.glob(f'{ctrl_f["data-path"][7:]}/*.out')
    
    raw_data = read_raw_data(out_files)
    raw_data_dict = raw_data_to_dict(raw_data)
    averaged_data = average_raw_data(raw_data_dict)

    Nint = len(raw_data[0])
    Tint = 1.0  # seconds
    t_int = np.arange(Nint) * Tint  # integration times

    def fit_pol_delay(pol):
        dels = []
        for i in range(Nint):
            lags_spec = raw_data_dict[0]["BL0BL1"][pol][i]
            corr = np.fft.irfft(lags_spec)
            corr = np.fft.fftshift(corr)
            lags = np.abs(corr)

            n = len(lags)
            t_lag = np.arange(-n//2, n//2 + 1)[:n]

            lag_idx = int(np.argmax(lags))
            lag_bins = t_lag[lag_idx]
            delay_seconds = lag_bins / BANDWIDTH_HZ
            dels.append(delay_seconds)

        dels = np.array(dels)
        coeff = np.polyfit(t_int, dels, 1)
        delay_rate = coeff[0]
        clock_offset = coeff[1]
        return delay_rate, clock_offset

    # ---- Fit RR and LL separately ----

    rr_rate, rr_offset = fit_pol_delay("RR")
    ll_rate, ll_offset = fit_pol_delay("LL")

    print("\nPer-polarization fits:")
    print(f"RR: delay_rate={rr_rate:.6e} s/s, clock_offset={rr_offset:.6e} s")
    print(f"LL: delay_rate={ll_rate:.6e} s/s, clock_offset={ll_offset:.6e} s")

    # ---- Average (since same delay applied to both pols) ----

    delay_rate_avg = 0.5 * (rr_rate + ll_rate)
    clock_offset_avg = 0.5 * (rr_offset + ll_offset)

    print("\nAveraged delay model:")
    print(f"delay_rate={delay_rate_avg:.6e} s/s")
    print(f"clock_offset={clock_offset_avg:.6e} s")
 
    normalized_data = {
        subband: {
            baseline: {
                pol: None for pol in POLARIZATIONS
            } for baseline in BASELINES
        } for subband in averaged_data.keys()
    }

    # for subband in averaged_data.keys():
    #     # global max amplitude over BL, pol, channels in this subband
    #     max_amp = 0.0
    #     for baseline in BASELINES:
    #         for pol in POLARIZATIONS:
    #             amp = np.abs(averaged_data[subband][baseline][pol])
    #             max_amp = max(max_amp, amp.max())
    #     for baseline in BASELINES:
    #         for pol in POLARIZATIONS:
    #             vis = averaged_data[subband][baseline][pol]
    #             normalized_data[subband][baseline][pol] = vis / max_amp



    # plot_data_single(normalized_data)
    # plot_data(normalized_data)

    plot_data_single(averaged_data)
    plot_data(averaged_data)