from utils.read_data import read_all, read_subband
from utils.process_data import weighted_mean, normalize_by_autos 
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

if __name__ == "__main__":
    output_path = "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/"
    output_files = (sorted(glob.glob(os.path.join(output_path, "*.out"))))

    headers, visibilities = read_all(output_files)
    processed_data = []

    for h, vis in zip(headers, visibilities):
        averaged = weighted_mean(h, vis)
        # normalized = normalize_by_autos(averaged)
        processed_data.append(averaged)

    processed_data = np.array(processed_data)
    print(processed_data.shape)
    processed_data = processed_data.reshape(processed_data.shape[0] * processed_data.shape[1], 3, 2, 2)
    print(processed_data.shape)

    data = {
            "XX": np.conj(processed_data[:, 1, 0, 0]),
            "XY": np.conj(processed_data[:, 1, 0, 1]),
            "YX": np.conj(processed_data[:, 1, 1, 0]),
            "YY": np.conj(processed_data[:, 1, 1, 1])
    }

    plt.figure(figsize=(12, 8))

    for pol in data:
        amp = np.abs(data[pol])
        phase = np.angle(data[pol], deg=True)
        x = np.arange(len(amp))
        corr = np.fft.irfft(data[pol])
        corr = np.fft.fftshift(corr)
        lags = np.arange(-len(corr) // 2, len(corr) // 2)[: len(corr)]

        plt.subplot(311)
        plt.title("Amplitude")
        plt.plot(x, amp, label=pol, linewidth=1)

        plt.subplot(312)
        plt.title("Phase")
        plt.scatter(x, phase, label=pol, s=5)

        plt.subplot(313)
        plt.title("Lag")
        plt.plot(lags, np.abs(corr), label=pol, linewidth=1)

    auto_data = {
            "XX": processed_data[:, 0, 0, 0],
            "YY": processed_data[:, 0, 1, 1],
    }

    plt.figure(figsize=(12, 8))

    for pol in auto_data:
        amp = np.abs(auto_data[pol])
        phase = np.angle(auto_data[pol], deg=True)
        x = np.arange(len(amp))

        plt.title("Amplitude")
        plt.plot(x, amp, label=pol, linewidth=1)

    auto_data = {
            "XX": processed_data[:, 2, 0, 0],
            "YY": processed_data[:, 2, 1, 1]
    }

    plt.figure(figsize=(12, 8))

    for pol in auto_data:
        amp = np.abs(auto_data[pol])
        phase = np.angle(auto_data[pol], deg=True)
        x = np.arange(len(amp))

        plt.title("Amplitude")
        plt.plot(x, amp, label=pol, linewidth=1)

    plt.show()

