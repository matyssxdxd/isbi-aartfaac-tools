import struct
import os
from collections.abc import Callable

import astropy.units as u
import numpy as np
from astropy.time import Time
from baseband import vdif
from tqdm import tqdm


class VDIFWriter:

    LEVELS_2BIT = np.array([-1, -1 / 3, 1 / 3, 1])

    def __init__(
        self,
        signal_func: Callable,
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

    def _quantize_2bit(self, data) -> np.ndarray:
        data = np.array(data)
        indices = np.abs(data[:, None] - self.LEVELS_2BIT).argmin(axis=1)
        return self.LEVELS_2BIT[indices]

    def _quantize(self, data) -> np.ndarray:
        match self.bps:
            case 2:
                return self._quantize_2bit(data)
            case _:
                raise ValueError(f"Unsupported bits per sample: {self.bps}. Supported values: 2")

    def _print_information(self) -> None:
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

    def run_cmd_das6(self, output_path) -> None:
        width = 100

        # Build the full command
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

        # Format delays to fit within bounds
        delays_0_str = str([f"{d}s" for d in delays_0])
        if len(delays_0_str) > 52:
            delays_0_str = delays_0_str[:49] + '...'
        print(f'{"Delays (Station 1):":<45} {delays_0_str:>54}')

        delays_1_str = str([f"{d}s" for d in delays_1])
        if len(delays_1_str) > 52:
            delays_1_str = delays_1_str[:49] + '...'
        print(f'{"Delays (Station 2):":<45} {delays_1_str:>54}')

        # Format center frequencies to fit within bounds
        center_freq_str = str([f"{freq * u.Hz.to(u.MHz):.1f} MHz" for freq in center_frequencies])
        if len(center_freq_str) > 52:
            center_freq_str = center_freq_str[:49] + '...'
        print(f'{"Subband center frequencies:":<45} {center_freq_str:>54}')

        # Format channel mapping to fit within bounds
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


def generate_sine_wave(freq, t, phase=0.0, pol=False) -> np.ndarray:
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
