import numpy as np

from .read_data import read_subband


def weighted_mean(headers: np.ndarray, visibilities: np.ndarray) -> np.ndarray:
    visibilities = np.asarray(visibilities, dtype=np.complex64)
    headers = np.asarray(headers, dtype=object)
    nr_baselines = visibilities.shape[-3]

    if visibilities.ndim == 5:
        weights = np.asarray(
            [header.weights[:nr_baselines] for header in headers],
            dtype=np.float32,
        )
        weights = weights[:, None, :, None, None]
        integration_axis = 0
    elif visibilities.ndim == 6:
        weights = np.asarray(
            [
                [header.weights[:nr_baselines] for header in subband_headers]
                for subband_headers in headers
            ],
            dtype=np.float32,
        )
        weights = weights[:, :, None, :, None, None]
        integration_axis = 1
    else:
        raise ValueError(
            "Expected visibilities with shape "
            "(integration, channel, baseline, 2, 2) or "
            "(subband, integration, channel, baseline, 2, 2), "
            f"got {visibilities.shape}"
        )

    weighted_sum = np.sum(visibilities * weights, axis=integration_axis)
    total_weight = np.sum(weights, axis=integration_axis)

    return np.divide(
        weighted_sum,
        total_weight,
        out=np.zeros_like(weighted_sum),
        where=total_weight > 0,
    )


def normalize_by_autos(visibilities: np.ndarray) -> np.ndarray:
    normalized = np.asarray(visibilities, dtype=np.complex64).copy()
    auto0 = np.maximum(np.real(normalized[..., 0, [0, 1], [0, 1]]), 1.0)
    auto1 = np.maximum(np.real(normalized[..., 2, [0, 1], [0, 1]]), 1.0)

    denom = np.sqrt(auto0[..., :, None] * auto1[..., None, :]).astype(np.float32)
    normalized[..., 1, :, :] = np.divide(
        normalized[..., 1, :, :],
        denom,
        out=np.zeros_like(normalized[..., 1, :, :]),
        where=denom > 0,
    )

    return normalized


if __name__ == "__main__":
    output_files = [
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_1.out",
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_2.out"
    ]

    headers, visibilities = read_subband(output_files[0])
    averaged_visibilities = weighted_mean(headers, visibilities)
    normalized_visibilities = normalize_by_autos(averaged_visibilities)
    print(normalized_visibilities.shape)
