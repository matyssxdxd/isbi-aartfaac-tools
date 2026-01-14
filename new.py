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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISBI-AARTFAAC correlator output processing')
    parser.add_argument(
            'input',
            nargs='+',
            help='One or more correlator output files, or a folder containig .out files'
    )
    parser.add_argument(
            'exper'
    )

    args = parser.parse_args()
    exper = args.exper

    output_files = extract_output_files(args.input) 
    headers, visibilities = read_all_visibilities(output_files)
    averaged_visibilities = average_visibilities(visibilities)

    cross_RR = averaged_visibilities[:, 1, :, 0]
    cross_RL = averaged_visibilities[:, 1, :, 1]
    cross_LR = averaged_visibilities[:, 1, :, 2]
    cross_LL = averaged_visibilities[:, 1, :, 3]

    all_cross = [cross_RR, cross_RL, cross_LR, cross_LL]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10)) 

    for i, cross in enumerate(all_cross):
        data = cross.flatten()
        x = np.linspace(6643.69, 6739.68, len(data))
        phase = np.angle(data, deg=True)
        ampl = np.abs(data)

        ax1.scatter(x, phase, label=POLS[i], s=6)
        ax1.set_ylabel('Phase (deg)')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylim(-200, 200)
        ax1.set_xlim(6643.69, 6739.68)
        ax1.legend()

        ax2.plot(x, ampl, label=POLS[i], linewidth=1)
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Frequency (MHz)')
        ax1.set_xlim(6643.69, 6739.68)
        ax2.legend()

        corr = np.fft.irfft(data)
        corr = np.fft.fftshift(corr)
        lags = np.arange(-len(corr) // 2, len(corr) // 2, + 1)[:len(corr)]

        ax3.plot(lags, np.abs(corr), label=POLS[i], linewidth=1)
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Lag')
        ax3.legend()


    plt.suptitle(f'{args.exper} | Amplitude + Phase + Lag')
    plt.tight_layout()
    plt.show()











