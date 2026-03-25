import numpy as np
import struct
from pathlib import Path
from typing import Tuple, List


class Header:
    def __init__(self):
        self.magic: int | None = None
        self.nr_receivers: int | None = None
        self.nr_polarizations: int | None = None
        self.correlation_mode: int | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.weights: Tuple[int, ...] | None = None
        self.nr_samples_per_integration: int | None = None
        self.nr_channels: int | None = None
        self.pad0: bytes | None = None
        self.first_channel_frequency: float | None = None
        self.channel_bandwidth: float | None = None
        self.pad1: bytes | None = None


def read_subband(output_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    headers = []
    visibilities = []

    with open(output_file, "rb") as file:
        while True:
            try:
                header = Header()
                header.magic = struct.unpack("I", file.read(4))[0]
                header.nr_receivers = struct.unpack("H", file.read(2))[0]
                header.nr_polarizations = struct.unpack("B", file.read(1))[0]
                header.correlation_mode = struct.unpack("B", file.read(1))[0]
                header.start_time = struct.unpack("d", file.read(8))[0]
                header.end_time = struct.unpack("d", file.read(8))[0]
                header.weights = struct.unpack("I" * 300, file.read(4 * 300))
                header.nr_samples_per_integration = struct.unpack("I", file.read(4))[0]
                header.nr_channels = struct.unpack("H", file.read(2))[0]
                header.pad0 = file.read(2)
                header.first_channel_frequency = struct.unpack("d", file.read(8))[0]
                header.channel_bandwidth = struct.unpack("d", file.read(8))[0]
                header.pad1 = file.read(288)

                assert header.nr_receivers is not None
                assert header.nr_channels is not None
                assert header.nr_polarizations is not None

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

                headers.append(header)
                visibilities.append(vis)

            except struct.error:
                break

    headers_array = np.asarray(headers, dtype=object)
    if visibilities:
        visibilities_array = np.asarray(visibilities)
    else:
        visibilities_array = np.empty((0,), dtype=np.complex64)

    return headers_array, visibilities_array


def read_all(output_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    all_headers = []
    all_visibilities = []

    for file in output_files:
        headers, visibilities = read_subband(file)
        all_headers.append(headers)
        all_visibilities.append(visibilities)

    return np.asarray(all_headers, dtype=object), np.asarray(
        all_visibilities, dtype=object
    )


if __name__ == "__main__":
    output_files = [
        Path(
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_1.out"
        ),
        Path(
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_1.out"
        ),
    ]

    subband_0_headers, subband_0_data = read_subband(output_files[0])
    print(subband_0_headers.shape)
    print(subband_0_data.shape)

    subband_headers, subband_data = read_all(output_files)
    print(subband_headers.shape)
    print(subband_data.shape)
