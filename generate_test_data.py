import numpy as np
from baseband import vdif
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

LEVELS = np.array([3, 1, -1, -3])

def generate_vdif_file(sample_rate, samples_per_frame, n_chan, n_thread, complex_data, bps,
                       edv, station, start_time, duration, base_freq, output_file, phase, delay):

    def generate_sine_wave(freq, t, phase=0.0, pol=False):
        if pol:
            sine_wave = np.sin(2 * np.pi * freq * t + np.pi / 2 + phase)
        else:
            sine_wave = np.sin(2 * np.pi * freq * t + phase)
        scaled_sine_wave = sine_wave * 3
        return scaled_sine_wave

    def quantize_2bit(samples):
        samples = np.array(samples)
        indices = np.abs(samples[:, None] - LEVELS).argmin(axis=1)
        return LEVELS[indices]

    fw = vdif.open(
        output_file, 'ws',
        sample_rate=sample_rate * u.Hz,
        samples_per_frame=samples_per_frame,
        nchan=n_chan,
        nthread=n_thread,
        complex_data=complex_data,
        bps=bps,
        edv=edv,
        station=station,
        time=start_time
    )

    n_frames = int(sample_rate * duration) // samples_per_frame

    for frame_idx in tqdm(range(n_frames), desc="Writing frames"):
        frame_data = np.zeros((samples_per_frame, n_chan), dtype=np.float32)
        t = (np.arange(samples_per_frame) + frame_idx * samples_per_frame) / sample_rate

        for freq_idx in range(n_chan // 2):
            freq = base_freq + freq_idx * base_freq

            sine_wave_rcp = generate_sine_wave(freq, (t + delay), phase, False)
            frame_data[:, 2 * freq_idx] = quantize_2bit(sine_wave_rcp)

            sine_wave_lcp = generate_sine_wave(freq, (t + delay), phase, True)
            frame_data[:, 2 * freq_idx + 1] = quantize_2bit(sine_wave_lcp)

        fw.write(frame_data)

    fw.close()

    print(f"VDIF file written to {output_file}")

if __name__ == "__main__":
    SAMPLE_RATE = 16e6
    SAMPLES_PER_FRAME = 2000
    N_THREADS = 1
    N_CHANNELS = 16
    COMPLEX_DATA = False
    BPS = 2
    EDV = 0
    STATION = 0
    DURATION = 20  # seconds
    OUTPUT_FILE_1 = "./sin_test_data_1_100kHz.vdif"
    OUTPUT_FILE_2 = "./sin_test_data_2_100kHz.vdif"
    START_TIME = Time("2025-09-30T10:10:10.0", format="isot", scale="utc")
    BASE_FREQ = 500e3 # kHz
    DELAY = 1e-6 # 1 microsecond delay

    generate_vdif_file(SAMPLE_RATE, SAMPLES_PER_FRAME, N_CHANNELS, N_THREADS, COMPLEX_DATA, BPS, EDV, STATION, START_TIME, DURATION, BASE_FREQ, OUTPUT_FILE_1, 0.0, 0.0);
    generate_vdif_file(SAMPLE_RATE, SAMPLES_PER_FRAME, N_CHANNELS, N_THREADS, COMPLEX_DATA, BPS, EDV, STATION, START_TIME, DURATION, BASE_FREQ, OUTPUT_FILE_2, 0.0, 0.0);


