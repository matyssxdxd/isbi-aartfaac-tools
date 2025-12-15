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

def read_raw_data(file_paths):
    raw_data = []
    for file in file_paths:
        header, visibilities = read_visibility_file(file)
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
            for pol in POLARIZATIONS:
                data = np.array(raw_data[subband][baseline][pol], dtype=np.complex64)
                avg_data = np.mean(data, axis=0)
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
                
            lags = np.abs(np.fft.fftshift(np.fft.irfft(data[subband]['BL0BL1'][pol])))
            n = len(lags)
            t = np.arange(-(n//2), n//2)
            lag = lags.argmax() - n + 1
            lag_offset = 1. - abs(lag)/float(n-1) if (abs(lag) < n - 1) else 1.
            # noise = 181 / (2 * 0.881 * np.sqrt(15000 * lag_offset))
            # snr = lags.max() / noise
            snr = 0
            lag_offset = 0
            
            if pol in ['RR', 'LL']:
                axs[0].set_title('Amplitude/Channel BL0 auto correlation')
                axs[0].plot(np.abs(data[subband]['BL0'][pol]), label='BL0')
                
                axs[1].set_title('Amplitude/Channel BL1 auto correlation')
                axs[1].plot(np.abs(data[subband]['BL1'][pol]), label='BL0')
                
                axs[2].set_title('Ampltiude/Lag cross-correlation')
                axs[2].plot(t, lags)
                
                axs[3].set_title('Amplitude/Channel cross-correlation')
                axs[3].plot(np.abs(data[subband]['BL0BL1'][pol]), label='BL0BL1')
                
                axs[4].set_title('Phase/Channel cross-correlation')
                axs[4].plot(np.angle(data[subband]['BL0BL1'][pol], deg=True), label='BL0BL1')
            else:
                axs[0].set_title('Ampltiude/Lag cross-correlation')
                axs[0].plot(t, lags)
                
                axs[1].set_title('Amplitude/Channel cross-correlation')
                axs[1].plot(np.abs(data[subband]['BL0BL1'][pol]), label='BL0BL1')
                
                axs[2].set_title('Phase/Channel cross-correlation')
                axs[2].plot(np.angle(data[subband]['BL0BL1'][pol], deg=True), label='BL0BL1')
                
            fig.suptitle(f'subband={subband}, pol={pol}')
            
            plt.tight_layout()
            plt.savefig(f'plot_subband_{subband}_pol_{pol}.png')
            plt.close()


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

    normalized_data = {
        subband: {
            baseline: {
                pol: None for pol in POLARIZATIONS
            } for baseline in BASELINES
        } for subband in averaged_data.keys()
    }

    for subband in averaged_data:
        for baseline in BASELINES:
            for pol in POLARIZATIONS:
                arr = np.array(averaged_data[subband][baseline][pol], dtype=np.complex64)

                amp = np.abs(arr)
                amp_min = amp.min()
                amp_max = amp.max()
                if amp_max > amp_min:
                    scale = (amp - amp_min) / (amp_max - amp_min)
                    norm = scale * np.exp(1j * np.angle(arr))
                else:
                    norm = arr

                normalized_data[subband][baseline][pol] = norm

    plot_data(averaged_data)

