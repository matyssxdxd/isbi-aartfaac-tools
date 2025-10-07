import numpy as np
from baseband import vdif
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

LEVELS = np.array([3, 1, -1, -3])

SAMPLE_RATE = 16e6
SAMPLES_PER_FRAME = 2000
N_THREADS = 1
N_CHANNELS = 16
COMPLEX_DATA = False
BPS = 2
EDV = 0
STATION = 0
DURATION = 20  # seconds
OUTPUT_FILE_1 = "./test_data_1.vdif"
OUTPUT_FILE_2 = "./test_data_2.vdif"
START_TIME = Time("2025-09-30T10:10:10.0", format="isot", scale="utc")

N_FRAMES = int(SAMPLE_RATE * DURATION) // SAMPLES_PER_FRAME
BASE_FREQ = 1e6 # MHz

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
    OUTPUT_FILE_1, 'ws',
    sample_rate=SAMPLE_RATE * u.Hz,
    samples_per_frame=SAMPLES_PER_FRAME,
    nchan=N_CHANNELS,
    nthread=N_THREADS,
    complex_data=COMPLEX_DATA,
    bps=BPS,
    edv=EDV,
    station=STATION,
    time=START_TIME
)

for frame_idx in tqdm(range(N_FRAMES), desc="Writing frames"):
    frame_data = np.zeros((SAMPLES_PER_FRAME, N_CHANNELS), dtype=np.float32)
    t = (np.arange(SAMPLES_PER_FRAME) + frame_idx * SAMPLES_PER_FRAME) / SAMPLE_RATE

    for freq_idx in range(N_CHANNELS // 2):
        freq = BASE_FREQ + freq_idx + 1

        sine_wave_rcp = generate_sine_wave(freq, t, False)
        frame_data[:, 2 * freq_idx] = quantize_2bit(sine_wave_rcp)

        sine_wave_lcp = generate_sine_wave(freq, t, True) 
        frame_data[:, 2 * freq_idx + 1] = quantize_2bit(sine_wave_lcp)

    fw.write(frame_data)

fw.close()

print(f"VDIF file written to {OUTPUT_FILE_1}")

fw = vdif.open(
    OUTPUT_FILE_2, 'ws',
    sample_rate=SAMPLE_RATE * u.Hz,
    samples_per_frame=SAMPLES_PER_FRAME,
    nchan=N_CHANNELS,
    nthread=N_THREADS,
    complex_data=COMPLEX_DATA,
    bps=BPS,
    edv=EDV,
    station=STATION,
    time=START_TIME
)

for frame_idx in tqdm(range(N_FRAMES), desc="Writing frames"):
    frame_data = np.zeros((SAMPLES_PER_FRAME, N_CHANNELS), dtype=np.float32)
    t = (np.arange(SAMPLES_PER_FRAME) + frame_idx * SAMPLES_PER_FRAME) / SAMPLE_RATE

    for freq_idx in range(N_CHANNELS // 2):
        freq = BASE_FREQ + freq_idx + 1

        sine_wave_rcp = generate_sine_wave(freq, t, 0.25, False)
        frame_data[:, 2 * freq_idx] = quantize_2bit(sine_wave_rcp)

        sine_wave_lcp = generate_sine_wave(freq, t, 0.25, True)
        frame_data[:, 2 * freq_idx + 1] = quantize_2bit(sine_wave_lcp)

    fw.write(frame_data)

fw.close()

print(f"VDIF file written to {OUTPUT_FILE_2}")
