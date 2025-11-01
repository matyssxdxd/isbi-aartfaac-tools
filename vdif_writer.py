"""VDIF Writer module for generating and writing VDIF format data files.

This module provides functionality to generate synthetic radio telescope data
and write it in VDIF (VLBI Data Interchange Format) format, commonly used
in radio astronomy for very long baseline interferometry.
"""

import struct
import os
from collections.abc import Callable
from typing import Any

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from astropy.time import Time
from baseband import vdif
from tqdm import tqdm


class VDIFWriter:
    """A class for generating and writing VDIF format data files.

    Attributes:
        LEVELS_2BIT: Quantization levels for 2-bit data encoding.
        signal_func: Callable function to generate signal data.
        sample_rate: Sampling rate in Hz.
        samples_per_frame: Number of samples per VDIF frame.
        n_chan: Number of channels.
        n_thread: Number of threads in VDIF format.
        complex_data: Whether data is complex valued.
        bps: Bits per sample (currently only 2 is supported).
        edv: Extended Data Version number.
        station: Station identifier.
        start_time: Start time of the observation.
        duration: Duration of the observation in seconds.
        frequency: Base frequency in Hz.
        frequency_spacing: Frequency spacing between subbands in Hz.
    """

    LEVELS_2BIT: NDArray[np.float64] = np.array([-1, -1 / 3, 1 / 3, 1])

    def __init__(
        self,
        signal_func: Callable[[float, NDArray[np.float64], float, bool], NDArray[np.float64]],
        sample_rate: float = 16e6,
        samples_per_frame: int = 2000,
        n_chan: int = 16,
        n_thread: int = 1,
        complex_data: bool = False,
        bps: int = 2,
        edv: int = 0,
        station: int = 0,
        start_time: Time = Time("2025-09-30T10:10:10.0",
                                format="isot", scale="utc"),
        duration: int = 20,
        frequency: float = 1e6,
        frequency_spacing: float = 0.0
    ) -> None:
        """Initialize the VDIFWriter with configuration parameters.

        Args:
            signal_func: Function that generates signal data. Should accept
                (frequency, time_array, phase, polarization) and return signal array.
            sample_rate: Sampling rate in Hz. Defaults to 16 MHz.
            samples_per_frame: Number of samples per VDIF frame. Defaults to 2000.
            n_chan: Number of channels. Defaults to 16.
            n_thread: Number of threads. Defaults to 1.
            complex_data: Whether to use complex data format. Defaults to False.
            bps: Bits per sample (only 2-bit supported). Defaults to 2.
            edv: Extended Data Version. Defaults to 0.
            station: Station identifier. Defaults to 0.
            start_time: Start time of observation. Defaults to "2025-09-30T10:10:10.0" UTC.
            duration: Duration of observation in seconds. Defaults to 20.
            frequency: Base frequency in Hz. Defaults to 1 MHz.
            frequency_spacing: Frequency spacing between subbands in Hz. Defaults to 0.0.
        """
        self.signal_func = signal_func
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.n_chan = n_chan
        self.n_thread = n_thread
        self.complex_data = complex_data
        self.bps = bps
        self.edv = edv
        self.station = station
        self.start_time = start_time
        self.duration = duration
        self.frequency = frequency
        self.frequency_spacing = frequency_spacing

        self._print_information()

    def _quantize_2bit(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantize data to 2-bit representation.

        Args:
            data: Input data array to be quantized.

        Returns:
            Quantized data array using 2-bit levels.
        """
        data = np.array(data)
        indices = np.abs(data[:, None] - self.LEVELS_2BIT).argmin(axis=1)
        return self.LEVELS_2BIT[indices]

    def _quantize(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantize data based on configured bits per sample.

        Args:
            data: Input data array to be quantized.

        Returns:
            Quantized data array.

        Raises:
            ValueError: If bps is not supported (currently only 2-bit is supported).
        """
        match self.bps:
            case 2:
                return self._quantize_2bit(data)
            case _:
                raise ValueError(f"Unsupported bits per sample: {self.bps}. Supported values: 2")

    def _print_information(self) -> None:
        """Print configuration information in a formatted table."""
        width = 100
        print('=' * width)
        print('VDIF Writer Information'.center(width))
        print('=' * width)
        print(f'{"Signal function:":<45} {self.signal_func.__name__:>54}')
        print(f'{"Sample rate:":<45} {str((self.sample_rate * u.Hz).to(u.MHz)):>54}')
        print(f'{"Samples per frame:":<45} {self.samples_per_frame:>54}')
        print(f'{"Number of channels:":<45} {self.n_chan:>54}')
        print(f'{"Number of threads:":<45} {self.n_thread:>54}')
        print(f'{"Complex data:":<45} {str(self.complex_data):>54}')
        print(f'{"Bits per sample:":<45} {self.bps:>54}')
        print(f'{"Extended data version:":<45} {self.edv:>54}')
        print(f'{"Station ID:":<45} {self.station:>54}')
        print(f'{"Start time:":<45} {str(self.start_time):>54}')
        print(f'{"Duration:":<45} {f"{self.duration}s":>54}')
        print(f'{"Signal base frequency:":<45} {str((self.frequency * u.Hz).to(u.MHz)):>54}')
        print(f'{"Signal frequency spacing:":<45} {str((self.frequency_spacing * u.Hz).to(u.MHz)):>54}')
        print('=' * width)

    def _plot_function(self, phase: float = 0.0, delay: float = 0.0) -> None:
        n_subbands = self.n_chan // 2
        fig, axs = plt.subplots(1, n_subbands)
        t = np.arange(self.samples_per_frame)

        for i in range(n_subbands):
            freq = self.frequency + i * self.frequency_spacing
            data_rcp = self.signal_func(freq, (t + delay), phase, False)
            data_lcp = self.signal_func(freq, (t + delay), phase, True)
            axs[i].plot(data_rcp)
            axs[i].plot(data_lcp)

        plt.show()

    def run_cmd_das6(self, output_path: str) -> None:
        """Print the command for running the correlator on DAS6 cluster.

        Args:
            output_path: Path to the output directory containing VDIF files.
        """
        width = 100

        cmd = (
            f'TZ=UTC ISBI/ISBI '
            f'--configFile /var/scratch/mpurvins/generated_data/{output_path}/data.conf '
            f'-n2 -t125000 -c128 -C127 -b16 -s8 -m15 '
            f'-D"2025-09-30 10:10:10" '
            f'-r{self.duration} '
            f'-g0 -q1 -R0 -B0 '
            f'-f{int(self.sample_rate)} '
            f'--subbandBandwidth 16e6 '
            f'-i /var/scratch/mpurvins/generated_data/{output_path}/data_1.vdif,'
            f'/var/scratch/mpurvins/generated_data/{output_path}/data_2.vdif '
            f'-o /var/scratch/mpurvins/{output_path}_1.out,'
            f'/var/scratch/mpurvins/{output_path}_2.out,'
            f'/var/scratch/mpurvins/{output_path}_3.out,'
            f'/var/scratch/mpurvins/{output_path}_4.out,'
            f'/var/scratch/mpurvins/{output_path}_5.out,'
            f'/var/scratch/mpurvins/{output_path}_6.out,'
            f'/var/scratch/mpurvins/{output_path}_7.out,'
            f'/var/scratch/mpurvins/{output_path}_8.out'
        )

        print('\n')
        print('=' * width)
        print('Correlator run cmd for DAS6'.center(width))
        print('=' * width)
        print(cmd)
        print('=' * width)

    def write(self, output_path: str, filename: str, phase: float = 0.0, delay: float = 0.0) -> None:
        """Write VDIF data file with generated signal data.

        Args:
            output_path: Relative path within './results/' directory for output.
            filename: Name of the VDIF file to create.
            phase: Phase offset to apply to the signal in radians. Defaults to 0.0.
            delay: Time delay to apply to the signal in seconds. Defaults to 0.0.
        """
        self._plot_function(phase, delay)

        full_path = f'./results/{output_path}'

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        width = 100
        print('\n')
        print('=' * width)
        print(f'Writing a VDIF file to: {full_path}/{filename}'.center(width))

        file = vdif.open(
            f'{full_path}/{filename}',
            "ws",
            sample_rate=self.sample_rate * u.Hz,
            samples_per_frame=self.samples_per_frame,
            nchan=self.n_chan,
            nthread=self.n_thread,
            complex_data=self.complex_data,
            bps=self.bps,
            edv=self.edv,
            station=self.station,
            time=self.start_time,
        )

        n_frames = int(self.sample_rate * self.duration) // self.samples_per_frame

        for frame in tqdm(range(n_frames), desc="", ncols=width, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            frame_data = np.zeros((self.samples_per_frame, self.n_chan), dtype=np.float32)
            t = (np.arange(self.samples_per_frame) + frame * self.samples_per_frame) / self.sample_rate

            for subband in range(self.n_chan // 2):
                freq = self.frequency + subband * self.frequency_spacing

                data = self.signal_func(freq, (t + delay), phase, False)
                frame_data[:, 2 * subband] = self._quantize(data)

                data = self.signal_func(freq, (t + delay), phase, True)
                frame_data[:, 2 * subband + 1] = self._quantize(data)

            file.write(frame_data)

        file.close()

        print(f'VDIF file with has been written to {full_path}/{filename}'.center(width))
        print('=' * width)

    def write_config(self, output_path: str, delay: float = 0.0) -> None:
        """Write binary configuration file for the correlator.

        Args:
            output_path: Relative path within './results/' directory for output.
            delay: Delay value to apply to station 2 in seconds. Defaults to 0.0.
                Station 1 will have zero delay.
        """
        width = 100
        mapping = list(range(self.n_chan))
        delay_count = self.duration // 2 + 1
        delays_0 = np.zeros(delay_count, dtype=np.double)
        delays_1 = np.full(delay_count, delay, dtype=np.double)
        delays = np.array([delays_0, delays_1], dtype=np.double)
        center_frequencies = [self.frequency * i for i in range(1, self.n_chan // 2 + 1)]

        print('\n')
        print('=' * width)
        print('Configuration file information'.center(width))
        print('=' * width)
        print(f'{"Delay count:":<45} {delay_count:>54}')

        delays_0_str = str([f"{d}s" for d in delays_0])
        if len(delays_0_str) > 52:
            delays_0_str = delays_0_str[:49] + '...'
        print(f'{"Delays (Station 1):":<45} {delays_0_str:>54}')

        delays_1_str = str([f"{d}s" for d in delays_1])
        if len(delays_1_str) > 52:
            delays_1_str = delays_1_str[:49] + '...'
        print(f'{"Delays (Station 2):":<45} {delays_1_str:>54}')

        center_freq_str = str([f"{freq * u.Hz.to(u.MHz):.1f} MHz" for freq in center_frequencies])
        if len(center_freq_str) > 52:
            center_freq_str = center_freq_str[:49] + '...'
        print(f'{"Subband center frequencies:":<45} {center_freq_str:>54}')

        mapping_str = str(mapping)
        if len(mapping_str) > 52:
            mapping_str = mapping_str[:49] + '...'
        print(f'{"Channel mapping:":<45} {mapping_str:>54}')

        print('=' * width)

        with open(f'./results/{output_path}/data.conf', "wb") as file:
            for d in delays:
                file.write(struct.pack("i", len(d)))
                file.write(struct.pack("d" * len(d), *d))

            file.write(struct.pack("i", len(center_frequencies)))
            file.write(struct.pack(
                "d" * len(center_frequencies), *center_frequencies))

            file.write(struct.pack("i", len(mapping)))
            file.write(struct.pack("i" * len(mapping), *mapping))

        print(f'Configuration file has been written to ./results/{output_path}/data.conf'.center(width))
        print('=' * width)


def generate_sine_wave(freq: float, t: NDArray[np.float64], phase: float = 0.0, pol: bool = False) -> NDArray[np.float64]:
    """Generate a sine wave signal.

    Args:
        freq: Frequency of the sine wave in Hz.
        t: Time array in seconds.
        phase: Phase offset in radians. Defaults to 0.0.
        pol: Polarization flag. If True, adds π/2 phase shift. Defaults to False.

    Returns:
        Array containing the generated sine wave values.
    """
    if pol:
        sine_wave = np.sin(2 * np.pi * freq * t + np.pi / 2 + phase)
    else:
        sine_wave = np.sin(2 * np.pi * freq * t + phase)
    return sine_wave

if __name__ == "__main__":
    vdif_writer = VDIFWriter(signal_func=generate_sine_wave, frequency=1.05e6, duration=20)
    vdif_writer.write(output_path='SIN_1_05MHZ', filename='data_1.vdif')
    vdif_writer.write(output_path='SIN_1_05MHZ', filename='data_2.vdif', delay=1e-6)
    vdif_writer.write_config(output_path='SIN_1_05MHZ', delay=1e-6)
    vdif_writer.run_cmd_das6('SIN_1_05MHZ')
