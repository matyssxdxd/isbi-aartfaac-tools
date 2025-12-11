import argparse
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils.process_data import process_data, read_visibility_file
import sys


class ISBIAARTFAACPlot:
    def __init__(self, filepaths, n_subbands):
        self.filepaths = filepaths
        self.n_subbands = n_subbands
        self.raw_data = []

        self.read_raw_data()

        self.data = {
            'SCALAR': {
                'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
            },
            'VECTOR': {
                'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
            },
            'INTEGRATION': {
                'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
            }
        }

        self.scalar_average()
        self.vector_average()
        self.integration_average()

    def read_raw_data(self, subband=None):
        if not subband:
            for file in self.filepaths:
                header, visibilities = read_visibility_file(file)
                self.raw_data.append(visibilities)
        else:
            file = self.filepaths[subband]
            header, visibilities = read_visibility_file(file)
            self.raw_data.append(visibilities)

    def scalar_average(self):
        for subband in range(self.n_subbands):
            raw_data = np.array(self.raw_data[subband], dtype=np.complex64)
            averaged_data = np.mean(raw_data, axis=0)
            averaged_data = np.swapaxes(averaged_data, 1, 2)
            for bidx, baseline in enumerate(['BL0', 'BL0BL1', 'BL1']):
                for pidx, pol in enumerate(['RR', 'RL', 'LR', 'LL']):
                    self.data['SCALAR'][baseline][pol].extend(
                        averaged_data[bidx][pidx])

    def vector_average(self):
        for subband in range(self.n_subbands):
            raw_data = np.array(self.raw_data[subband], dtype=np.complex64)
            vec_averaged_data = np.sum(raw_data, axis=0)
            vec_averaged_data = np.swapaxes(vec_averaged_data, 1, 2)
            for bidx, baseline in enumerate(['BL0', 'BL0BL1', 'BL1']):
                for pidx, pol in enumerate(['RR', 'RL', 'LR', 'LL']):
                    self.data['VECTOR'][baseline][pol].extend(
                        vec_averaged_data[bidx][pidx])

    def integration_average(self):
        result = []
        for subband in range(self.n_subbands):
            raw_data = np.array(self.raw_data[subband], dtype=np.complex64)
            for vis in raw_data:
                int_averaged_data = np.mean(vis, axis=1)
                for bidx, baseline in enumerate(['BL0', 'BL0BL1', 'BL1']):
                    for pidx, pol in enumerate(['RR', 'RL', 'LR', 'LL']):
                        self.data['INTEGRATION'][baseline][pol].append(
                            int_averaged_data[bidx][pidx])

    def show(self, plot_name, vector_average=False):
        if vector_average:
            averaging = 'VECTOR'
        else:
            averaging = 'SCALAR'

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, sharex=True, figsize=(12, 8))

        x_axis = np.arange(len(self.data[averaging]['BL0']['RR']))
        for pol in ['RR', 'RL', 'LL', 'LR']:
            if pol in ['RR', 'LL']:
                ax1.plot(x_axis, np.abs(
                    self.data[averaging]['BL0'][pol]), label=pol)
                ax2.plot(x_axis, np.abs(
                    self.data[averaging]['BL1'][pol]), label=pol)

            ax3.scatter(x_axis, np.angle(
                self.data[averaging]['BL0BL1'][pol], deg=True), label=pol, s=10)
            ax4.plot(x_axis, np.abs(
                self.data[averaging]['BL0BL1'][pol]), label=pol)


        ax1.set_title('AUTO BL0')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)

        ax2.set_ylabel('Amplitude')
        ax2.set_title('AUTO BL1')
        ax2.legend()
        ax2.grid(True)

        ax3.set_title('CROSS BL0BL1')
        ax3.set_ylabel('Phase (deg)')
        ax3.set_xlabel('Output channel')
        ax3.legend()
        ax3.grid(True)

        ax4.set_title('CROSS BL0BL1')
        ax4.set_ylabel('Amplitude')
        ax4.set_xlabel('Output channel')
        ax4.legend()
        ax4.grid(True)

        plt.suptitle(plot_name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ctrl", help="Control file")

    args = parser.parse_args()

    with open(args.ctrl, 'r') as ctrl:
        ctrl_f = json.load(ctrl)

    out_files = glob.glob(f'{ctrl_f["data-path"][7:]}/*.out')
    n_subbands = len(ctrl_f['subbands'])
    print(n_subbands)

    plot = ISBIAARTFAACPlot(out_files, n_subbands)
    plot.show(ctrl_f['plot-name'], vector_average=True)

    # integration_time = 2  # seconds
    # num_integrations = len(rr_arr) // 8

    # time = np.arange(num_integrations) * integration_time
    # time_full = np.tile(time, 8)
    # x = np.arange(len(rr_arr)) % num_integrations * integration_time

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    # fig.suptitle(ctrl_f['plot-name'])
    # ax1.scatter(x, np.angle(rr_arr, deg=True), label='RR', s=8)
    # # ax1.scatter(x, np.angle(rl_arr, deg=True), label='RL', s=8)
    # # ax1.scatter(x, np.angle(lr_arr, deg=True), label='LR', s=8)
    # ax1.scatter(x, np.angle(ll_arr, deg=True), label='LL', s=8)
    # ax1.set_ylim([-200, 200])
    # ax1.set_ylabel('Phase (deg)')
    # ax1.legend()
    # ax1.grid(True)

    # ax2.plot(x, np.abs(rr_arr), label='RR')
    # # ax2.plot(x, np.abs(rl_arr), label='RL')
    # # ax2.plot(x, np.abs(lr_arr), label='LR')
    # ax2.plot(x, np.abs(ll_arr), label='LL')
    # ax2.set_ylim([-0.1 * 1e8, 2 * 1e8])
    # ax2.set_ylabel('Amplitude')
    # ax2.set_xlabel('Time (s)')
    # ax2.legend()
    # ax2.grid(True)

    # plt.tight_layout()
    # plt.show()
