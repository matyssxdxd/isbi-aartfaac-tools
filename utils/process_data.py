import matplotlib.pyplot as plt
import numpy as np

from utils.read_data import read_subband


def extract_weights(headers, n_baselines=3):
    w = np.asarray([h.weights[:n_baselines] for h in headers], dtype=np.float64)
    return w


def weighted_average_integrations(data, weights):
    data = np.asarray(data)
    weights = np.asarray(weights, dtype=np.float64)

    # weights: (integration, baseline) -> broadcast to data
    w = weights[:, None, :, None, None]

    weighted_sum = np.sum(w * data, axis=0)
    weight_sum = np.sum(w, axis=0)

    avg = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum, dtype=np.result_type(data, np.complex64)),
        where=weight_sum > 0
    )

    return avg, weight_sum


def normalize_cross(data, auto0_bl=0, cross_bl=1, auto1_bl=2, eps=1e-12):
    data = np.asarray(data)
    data_norm = data.copy()

    # data shape: (integration, channel, baseline, polx, poly)

    a0_xx = np.real(data[:, :, auto0_bl, 0, 0])
    a0_yy = np.real(data[:, :, auto0_bl, 1, 1])
    a1_xx = np.real(data[:, :, auto1_bl, 0, 0])
    a1_yy = np.real(data[:, :, auto1_bl, 1, 1])

    denom = np.empty((data.shape[0], data.shape[1], 2, 2), dtype=np.float64)

    denom[:, :, 0, 0] = np.sqrt(np.maximum(a0_xx * a1_xx, eps))
    denom[:, :, 0, 1] = np.sqrt(np.maximum(a0_xx * a1_yy, eps))
    denom[:, :, 1, 0] = np.sqrt(np.maximum(a0_yy * a1_xx, eps))
    denom[:, :, 1, 1] = np.sqrt(np.maximum(a0_yy * a1_yy, eps))

    data_norm[:, :, cross_bl, :, :] = data[:, :, cross_bl, :, :] / denom

    return data_norm

def normalize_auto(data, eps=1e-12):
    data = np.array(data)
    out = data.copy()
    refs = {}

    for bl in (0, 2):
        for p in range(2):
            ref = np.mean(np.real(data[:, :, bl, p, p]))
            ref = max(ref, eps)
            out[:, :, bl, p, p] = data[:, :, bl, p, p] / ref
            refs[(bl, p, p)] = ref

    return out, refs

if __name__ == "__main__":
    output_files = [
        "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_1.out",
        "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_2.out"
    ]

    headers, visibilities = read_subband(output_files[0])
    print("visibilities shape:", visibilities.shape)

    weights = extract_weights(headers)
    print("weights shape:", weights.shape)

    norm = normalize_cross(visibilities)
    norm, refs = normalize_auto(norm)
    print("norm shape:", norm.shape)

    avg, avg_w = weighted_average_integrations(norm, weights)
    print("time-averaged shape:", avg.shape)

    avg = np.conj(avg)

    plt.figure(figsize=(12, 8))

    for polx in range(2):
        for poly in range(2):
            x = np.arange(len(avg))
            plt.subplot(211)
            plt.plot(x, np.abs(avg[:, 1, polx, poly]))

            plt.subplot(212)
            phase = np.angle(avg[:, 1, polx, poly], deg=True)
            plt.scatter(x, phase, s=3, label=f"{polx}-{poly}")

    plt.figure(figsize=(12, 8))

    x = np.arange(len(avg))
    plt.subplot(211)
    plt.plot(x, np.abs(avg[:, 0, 0, 0]))
    plt.plot(x, np.abs(avg[:, 0, 1, 1]))

    plt.subplot(212)
    plt.plot(x, np.abs(avg[:, 2, 0, 0]))
    plt.plot(x, np.abs(avg[:, 2, 1, 1]))

    plt.show()
