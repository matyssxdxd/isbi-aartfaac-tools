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

        # Changed from 2x2 to 3x2 to accommodate Lag Plot (ax5) and Channel Zoom (ax6)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            3, 2, sharex=False, figsize=(12, 12)) # Removed sharex=True because Lag plot has different X-axis

        # X-axis for frequency plots
        x_axis_freq = np.arange(len(self.data[averaging]['BL0']['RR']))

        # Identify the peak channel for the Zoom plot (simple max finder on RR cross)
        # We use the vector averaged data for finding the peak to avoid noise
        cross_spectrum_rr = np.array(self.data['VECTOR']['BL0BL1']['RR'])
        peak_channel_idx = np.argmax(np.abs(cross_spectrum_rr))
        print(f"Detected peak at channel: {peak_channel_idx}")

        for pol in ['RR', 'RL', 'LR', 'LL']:
            if pol in ['RR', 'LL']: # Plot Autocorrs
                ax1.plot(x_axis_freq, np.abs(self.data[averaging]['BL0'][pol]), label=pol)
                ax2.plot(x_axis_freq, np.abs(self.data[averaging]['BL1'][pol]), label=pol)

            # Cross Phase & Amplitude (Full Band)
            ax3.scatter(x_axis_freq, np.angle(self.data[averaging]['BL0BL1'][pol], deg=True), label=pol, s=10)
            ax4.plot(x_axis_freq, np.abs(self.data[averaging]['BL0BL1'][pol]), label=pol)

            # --- New Plot 1: Lag Space (IFFT of Cross Spectrum) ---
            # We take the IFFT of the complex visibility
            # Shift zero-frequency component to center of spectrum before IFFT? Usually just IFFT.
            # FFTshift after IFFT to put 0 lag in center.
            complex_vis = np.array(self.data[averaging]['BL0BL1'][pol])
            lag_spectrum = np.fft.fftshift(np.fft.ifft(complex_vis))
            lag_axis = np.arange(-len(lag_spectrum)//2, len(lag_spectrum)//2)

            ax5.plot(lag_axis, np.abs(lag_spectrum), label=pol)

            # --- New Plot 2: Time Series of Peak Channel (Phase) ---
            # We need the time-series data, which is stored in INTEGRATION dictionary
            # But INTEGRATION stores averaged-over-channels data? 
            # Wait, original code: int_averaged_data = np.mean(vis, axis=1) -> averages over channels (axis 1)
            # So 'INTEGRATION' dict loses frequency resolution. We cannot use it for single-channel time series.
            # We must go back to raw_data to extract the time evolution of the specific channel.

            # Re-extract time series for the peak channel
            # raw_data structure: [subband][time, channel, baseline, pol] (based on typical shapes, implied by np.mean(vis, axis=1))
            # Actually, let's look at scalar_average: raw_data is [subband] -> np.array -> shape?
            # averaged_data = np.mean(raw_data, axis=0) -> axis 0 is time.
            # So raw_data[subband] is (Time, Freq_Channels, Baselines, Pols).

            # Let's aggregate time series for the specific channel across subbands? 
            # Usually 'subbands' are concatenated in frequency or just separate chunks.
            # Assuming we just want to see the time evolution of the peak channel found in the integrated spectrum.
            # We will concat all time points from all subbands (if they are time-chunks) or just the first subband?
            # The original code loops subbands and extends lists...
            # Let's assume subbands are frequency chunks. We need to find WHICH subband the peak is in.

            # Actually, the 'data' dict extends lists across subbands. 
            # So x_axis_freq covers all subbands concatenated.
            # We need to map peak_channel_idx back to (subband_idx, channel_inside_subband).

            pass # Logic implemented below outside the loop to avoid re-calculating

        # --- Logic for Channel Time Series ---
        # 1. Find global channel mapping
        # We assume all subbands have equal number of channels.
        # We need to know channels per subband.
        # We can infer it from the first subband's raw_data shape.

        # But we can't easily access 'raw_data' here because it might be heavy/cleared? 
        # The class keeps 'self.raw_data'.
        if self.raw_data:
            sample_subband = np.array(self.raw_data[0], dtype=np.complex64)
            # Shape: (Time, Channels, Baselines, Pols)
            n_channels_per_subband = sample_subband.shape[1]

            target_subband = peak_channel_idx // n_channels_per_subband
            target_channel_local = peak_channel_idx % n_channels_per_subband

            if target_subband < len(self.raw_data):
                 # Extract time series for this specific channel
                vis_data = np.array(self.raw_data[target_subband], dtype=np.complex64)
                # vis_data shape: (Time, Channels, Baselines, Pols)
                # We want Cross Baseline (Index 1 usually, based on 'BL0', 'BL0BL1', 'BL1')
                # In scalar_average: enumerate(['BL0', 'BL0BL1', 'BL1']) -> bidx 1 is BL0BL1

                time_series_complex = vis_data[:, target_channel_local, 1, :] # Shape (Time, Pols)

                # Plot Phase over time for RR (idx 0) and LL (idx 3)
                # Pol indices: 0=RR, 1=RL, 2=LR, 3=LL
                times = np.arange(len(time_series_complex))

                ax6.scatter(times, np.angle(time_series_complex[:, 0], deg=True), label='RR', s=5, alpha=0.7)
                ax6.scatter(times, np.angle(time_series_complex[:, 3], deg=True), label='LL', s=5, alpha=0.7)

        # Formatting
        ax1.set_title('AUTO BL0')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True)

        ax2.set_title('AUTO BL1')
        ax2.set_ylabel('Amplitude')
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(True)

        ax3.set_title('CROSS BL0BL1 Phase')
        ax3.set_ylabel('Phase (deg)')
        # ax3.set_xlabel('Output channel') # Shared with ax4? No, sharex=False now.
        ax3.legend(loc='upper right', fontsize='small')
        ax3.grid(True)

        ax4.set_title('CROSS BL0BL1 Amp')
        ax4.set_ylabel('Amplitude')
        # ax4.set_xlabel('Output channel')
        ax4.legend(loc='upper right', fontsize='small')
        ax4.grid(True)

        ax5.set_title('LAG PLOT (IFFT of Cross)')
        ax5.set_xlabel('Lag (samples)')
        ax5.set_ylabel('Amplitude')
        ax5.legend(loc='upper right', fontsize='small')
        ax5.grid(True)

        ax6.set_title(f'Phase vs Time (Peak Ch {peak_channel_idx})')
        ax6.set_xlabel('Time Integration #')
        ax6.set_ylabel('Phase (deg)')
        ax6.set_ylim(-180, 180)
        ax6.legend(loc='upper right', fontsize='small')
        ax6.grid(True)

        plt.suptitle(plot_name)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ctrl", help="Control file")
    args = parser.parse_args()

    with open(args.ctrl, 'r') as ctrl:
        ctrl_f = json.load(ctrl)

    # Support absolute paths or relative paths
    out_files = glob.glob(f'{ctrl_f["data-path"][7:]}/*.out')
    if not out_files:
         # Try direct path if the slicing [7:] (for 'file://') isn't needed or correct
         out_files = glob.glob(f'{ctrl_f["data-path"]}/*.out')

    n_subbands = len(ctrl_f['subbands'])
    print(f"Found {len(out_files)} files for {n_subbands} subbands specified.")

    # Ensure we don't crash if file counts mismatch, but warn
    if len(out_files) == 0:
        print("No .out files found! Check data-path.")
        sys.exit(1)

    plot = ISBIAARTFAACPlot(out_files, n_subbands)
    plot.show(ctrl_f['plot-name'], vector_average=True)
