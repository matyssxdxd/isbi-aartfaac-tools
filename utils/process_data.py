import numpy as np
import struct


class Header:
    def __init__(self):
        self.magic = None
        self.nr_receivers = None
        self.nr_polarizations = None
        self.correlation_mode = None
        self.start_time = None
        self.end_time = None
        self.weights = None
        self.nr_samples_per_integration = None
        self.nr_channels = None
        self.pad0 = None
        self.first_channel_frequency = None
        self.channel_bandwidth = None
        self.pad1 = None


def _clamped_power(auto_spectrum):
    power = float(np.mean(np.real(auto_spectrum)))
    return power if power >= 1.0 else 1.0


def normalize_visibility_block(vis):
    """Apply SFXC-like correlation normalization for a 2-receiver visibility block.

    Expected baseline layout for 2 receivers is [auto0, cross01, auto1] and
    polarization order [RR, RL, LR, LL].
    """
    vis_norm = np.array(vis, dtype=np.complex64, copy=True)

    if vis_norm.ndim != 3:
        return vis_norm

    n_baselines, _, n_pols = vis_norm.shape
    if n_baselines < 3 or n_pols < 4:
        return vis_norm

    auto0_rr = _clamped_power(vis_norm[0, :, 0])
    auto0_ll = _clamped_power(vis_norm[0, :, 3])
    auto1_rr = _clamped_power(vis_norm[2, :, 0])
    auto1_ll = _clamped_power(vis_norm[2, :, 3])

    cross_denoms = np.array(
        [
            np.sqrt(auto0_rr * auto1_rr),
            np.sqrt(auto0_rr * auto1_ll),
            np.sqrt(auto0_ll * auto1_rr),
            np.sqrt(auto0_ll * auto1_ll),
        ],
        dtype=np.float32,
    )
    auto0_denoms = np.array(
        [
            auto0_rr,
            np.sqrt(auto0_rr * auto0_ll),
            np.sqrt(auto0_ll * auto0_rr),
            auto0_ll,
        ],
        dtype=np.float32,
    )
    auto1_denoms = np.array(
        [
            auto1_rr,
            np.sqrt(auto1_rr * auto1_ll),
            np.sqrt(auto1_ll * auto1_rr),
            auto1_ll,
        ],
        dtype=np.float32,
    )

    vis_norm[0, :, :4] /= auto0_denoms[np.newaxis, :]
    vis_norm[1, :, :4] /= cross_denoms[np.newaxis, :]
    vis_norm[2, :, :4] /= auto1_denoms[np.newaxis, :]
    return vis_norm


def normalize_visibilities(visibilities):
    return [normalize_visibility_block(vis) for vis in visibilities]


def read_visibility_file(visibility_path, normalize=False):
    headers = []
    visibilities = []

    with open(visibility_path, 'rb') as file:
        while True:
            try:
                # Read header
                header = Header()
                header.magic = struct.unpack('I', file.read(4))[0]
                header.nr_receivers = struct.unpack('H', file.read(2))[0]
                header.nr_polarizations = struct.unpack('B', file.read(1))[0]
                header.correlation_mode = struct.unpack('B', file.read(1))[0]
                header.start_time = struct.unpack('d', file.read(8))[0]
                header.end_time = struct.unpack('d', file.read(8))[0]
                header.weights = struct.unpack('I' * 300, file.read(4 * 300))
                header.nr_samples_per_integration = struct.unpack('I', file.read(4))[0]
                header.nr_channels = struct.unpack('H', file.read(2))[0]
                header.pad0 = file.read(2)
                header.first_channel_frequency = struct.unpack('d', file.read(8))[0]
                header.channel_bandwidth = struct.unpack('d', file.read(8))[0]
                header.pad1 = file.read(288)

                vis_dtype = np.complex64
                nr_baselines = header.nr_receivers + int(
                    header.nr_receivers * (header.nr_receivers - 1) / 2
                )
                vis_shape = (nr_baselines, header.nr_channels, header.nr_polarizations)
                vis_zeros = np.zeros(vis_shape, vis_dtype)

                # Read visibilities
                vis = file.read(vis_zeros.size * vis_zeros.itemsize)
                if len(vis) < vis_zeros.size * vis_zeros.itemsize:
                    break

                vis = np.frombuffer(vis, dtype=vis_dtype).reshape(vis_shape)
                if normalize:
                    vis = normalize_visibility_block(vis)

                headers.append(header)
                visibilities.append(vis)

            except struct.error:
                break

    return headers, visibilities


def average_visibilities(visibilities):
    vis_array = np.array(visibilities, dtype=np.complex64)
    averaged_visibilities = np.mean(vis_array, axis=0)
    averaged_visibilities = np.swapaxes(averaged_visibilities, 1, 2)

    return averaged_visibilities


def vector_sum_visibilities(visibilities):
    vis_array = np.array(visibilities, dtype=np.complex64)
    summed_visibilities = np.sum(vis_array, axis=0)
    vector_average = summed_visibilities
    vector_average = np.swapaxes(vector_average, 1, 2)
    return vector_average


def average_integrations(visibilities):
    result = []
    for vis in visibilities:
        avg = np.mean(vis, axis=1)
        result.append(avg)
    return np.array(result, dtype=np.complex64)


def process_data(corr_files, integration_average=False, vector_average=False, normalize=False):
    processed_visibilities = []

    for file in corr_files:
        _, visibilities = read_visibility_file(file, normalize=normalize)
        if integration_average:
            processed_visibilities.append(average_integrations(visibilities))
        elif vector_average:
            processed_visibilities.append(vector_sum_visibilities(visibilities))
        else:
            processed_visibilities.append(average_visibilities(visibilities))

    return processed_visibilities