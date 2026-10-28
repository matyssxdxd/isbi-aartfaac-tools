import matplotlib.pyplot as plt
import numpy as np

from utils.read_data import read_subband, nr_baselines, baseline_index


def extract_weights(headers, n_baselines=None):
    """Per-integration, per-baseline weights, shape (integration, baseline).

    `n_baselines` defaults to what the header says (3 for two stations, 6 for
    three, ...). Negative weights -- a baseline with no valid samples -- and
    the zeros --referenceStation writes for baselines it did not produce both
    clip to 0, so they drop out of a weighted average instead of dominating it.
    """
    if n_baselines is None:
        n_baselines = nr_baselines(headers[0].nr_receivers)
    w = np.asarray([h.weights[:n_baselines] for h in headers], dtype=np.float64)
    return np.clip(w, 0, None)


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


def _nr_stations(data):
    """Stations in a (integration, channel, baseline, pol, pol) array."""
    nbl = data.shape[2]
    return int(round((np.sqrt(8 * nbl + 1) - 1) / 2))


def normalize_cross(data, eps=1e-12):
    """Every cross baseline divided by sqrt(auto_i * auto_j), per channel and
    polarisation pair, giving a coherence. Autocorrelations are left as they
    are. Works for any number of stations.

    Note this is not just a rescaling: dividing per channel removes the
    bandpass, so the SHAPE of the cross spectrum changes too.
    """
    data = np.asarray(data)
    data_norm = data.copy()

    # data shape: (integration, channel, baseline, polx, poly)
    n = _nr_stations(data)
    for s2 in range(n):
        for s1 in range(s2):
            a1 = np.real(data[:, :, baseline_index(s1, s1)])
            a2 = np.real(data[:, :, baseline_index(s2, s2)])
            denom = np.empty((data.shape[0], data.shape[1], 2, 2), dtype=np.float64)
            for px in range(2):
                for py in range(2):
                    denom[:, :, px, py] = np.sqrt(np.maximum(a1[:, :, px, px] * a2[:, :, py, py], eps))
            bl = baseline_index(s1, s2)
            data_norm[:, :, bl, :, :] = data[:, :, bl, :, :] / denom

    return data_norm

def normalize_auto(data, eps=1e-12):
    data = np.array(data)
    out = data.copy()
    refs = {}

    for bl in (baseline_index(s, s) for s in range(_nr_stations(data))):
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
