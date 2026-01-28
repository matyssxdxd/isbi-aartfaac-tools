#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os

from utils.process_data import read_visibility_file

BL_AUTO0 = 0
BL_CROSS = 1
BL_AUTO1 = 2
FREQ_LOW = 6675.69 - 16.0
FREQ_HIGH = 6675.69

POLS = ['RR', 'RL', 'LR', 'LL']

def extract_output_files(input_paths):
    output_files = []

    for path in input_paths:
        if os.path.isdir(path):
            out_files = glob.glob(os.path.join(path, '*.out'))
            output_files.extend(sorted(out_files))
        elif os.path.isfile(path):
            output_files.append(path)
        else:
            print(f'Warning: \'{path}\' is not a valid file or directory')

    return output_files

def read_all_visibilities(output_files):
    all_headers = []
    all_visibilities = []

    for file in output_files:
        print(f'Reading: {file}')
        headers, visibilities = read_visibility_file(file)
        all_headers.append(headers)
        all_visibilities.append(visibilities)
        print(f' - {len(headers)} integrations, shape: {visibilities[0].shape}')

    return np.array(all_headers), np.array(all_visibilities)

# TODO: Might be better to implement different types of averaging
def average_visibilities(visibilities):
    averaged_visibilities = []

    for vis in visibilities:
        averaged_visibilities.append(np.mean(vis, axis=0))

    return np.array(averaged_visibilities)

def plot(input, exper, flip=False, integration=None):
    output_files = extract_output_files(input)
    headers, visibilities = read_all_visibilities(output_files)
    n_integrations = visibilities.shape[1]

    print(visibilities.shape)

    if integration is not None:
        if integration < 0 or integration >= n_integrations:
            raise ValueError(f'Integration {integration} out of range (0-{n_integrations-1})')
        selected_visibilities = visibilities[:, integration, :, :, :]
        title_suffix = f' | Integration {integration}'
    else:
        selected_visibilities = average_visibilities(visibilities)
        title_suffix = ' | Averaged'

    print(f'selected_visibilities.shape: {selected_visibilities.shape}')
    cross_RR = selected_visibilities[:, 1, :, 0]
    cross_RL = selected_visibilities[:, 1, :, 1]
    cross_LR = selected_visibilities[:, 1, :, 2]
    cross_LL = selected_visibilities[:, 1, :, 3]
    all_cross = [cross_RR, cross_RL, cross_LR, cross_LL]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10)) 

    for i, cross in enumerate(all_cross):
        data = cross.flatten()
        if flip:
            data = np.flip(data)
        x = np.linspace(FREQ_LOW, FREQ_HIGH, len(data))
        phase = np.angle(data, deg=True)
        ampl = np.abs(data)

        ax1.scatter(x, phase, label=POLS[i], s=6)
        ax1.set_ylabel('Phase (deg)')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylim(-200, 200)
        ax1.legend()

        ax2.plot(x, ampl, label=POLS[i], linewidth=1)
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Frequency (MHz)')
        ax2.legend()

        corr = np.fft.irfft(data)
        corr = np.fft.fftshift(corr)
        lags = np.arange(-len(corr) // 2, len(corr) // 2, + 1)[:len(corr)]
        peak_idx = np.argmax(np.abs(corr))
        lag_peak = lags[peak_idx]
        print(lag_peak, POLS[i])

        ax3.plot(lags, np.abs(corr), label=POLS[i], linewidth=1)
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Lag')
        ax3.legend()

    plt.suptitle(f'{exper} | Phase + Amplitude + Lag{title_suffix}')
    plt.tight_layout()
    plt.show()

    from sfxcdata import SFXCData

    sfxc = SFXCData('./corr_files/B023.cor_0002')
    sfxc.next_integration()

    vis = ('Ib', 'Ir')
    cross_chans = sfxc.vis[vis].keys()
    chans = []

    integrations_dict = {
        'RR': [],
        'RL': [],
        'LR': [],
        'LL': []
    }

    while sfxc.next_integration():
        cross_chans = sfxc.vis[vis].keys()
        for chan in cross_chans:
            if chan.freqnr == 1 and chan.sideband == 0:
                data = sfxc.vis[vis][chan].vis
                if chan.pol1 == 0 and chan.pol2 == 0:
                    integrations_dict['RR'].append(data)
                elif chan.pol1 == 0 and chan.pol2 == 1:
                    integrations_dict['RL'].append(data)
                elif chan.pol1 == 1 and chan.pol2 == 0:
                    integrations_dict['LR'].append(data)
                elif chan.pol1 == 1 and chan.pol2 == 1:
                    integrations_dict['LL'].append(data)

    if integration is not None:
        n_integrations = len(integrations_dict['RR'])
        if integration < 0 or integration >= n_integrations:
            raise ValueError(f'Integration {integration} out of range (0-{n_integrations-1})')
        for pol in integrations_dict:
            integrations_dict[pol] = integrations_dict[pol][integration]
        sfxc_title_suffix = f' | Integration {integration}'
    else:
        for pol in integrations_dict:
            integrations_dict[pol] = np.mean(integrations_dict[pol], axis=0)
        sfxc_title_suffix = ' | Averaged'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    all_cross = [
        np.array(integrations_dict['RR']),
        np.array(integrations_dict['RL']),
        np.array(integrations_dict['LR']),
        np.array(integrations_dict['LL'])
    ]

    for i, cross in enumerate(all_cross):
        data = np.flip(cross.flatten())
        x = np.linspace(FREQ_LOW, FREQ_HIGH, len(data))
        phase = np.angle(data, deg=True)
        ampl = np.abs(data)

        ax1.scatter(x, phase, label=POLS[i], s=6)
        ax1.set_ylabel('Phase (deg)')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylim(-200, 200)
        ax1.legend()

        ax2.plot(x, ampl, label=POLS[i], linewidth=1)
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Frequency (MHz)')
        ax2.legend()

        corr = np.fft.irfft(data)
        corr = np.fft.fftshift(corr)
        lags = np.arange(-len(corr) // 2, len(corr) // 2, + 1)[:len(corr)]

        peak_idx = np.argmax(np.abs(corr))
        lag_peak = lags[peak_idx]
        print(lag_peak, POLS[i])

        ax3.plot(lags, np.abs(corr), label=POLS[i], linewidth=1)
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Lag')
        ax3.legend()

    plt.suptitle(f'{exper} | Phase + Amplitude + Lag (SFXCData){sfxc_title_suffix}')
    plt.tight_layout()
    plt.show()
