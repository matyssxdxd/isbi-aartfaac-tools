import numpy as np
import struct

from dataclasses import dataclass
from typing import Tuple, List

@dataclass(frozen=True)
class Header:
    magic: int
    nr_receivers: int
    nr_polarizations: int
    correlation_mode: int
    start_time: float
    end_time: float
    weights: Tuple[int, ...]
    nr_samples_per_integration: int
    nr_channels: int
    first_channel_frequency: float
    channel_bandwidth: float

def nr_baselines(nr_receivers: int) -> int:
    """Baselines in a .out record, autocorrelations included."""
    return nr_receivers * (nr_receivers + 1) // 2


def baseline_index(station1: int, station2: int) -> int:
    """Index of baseline (station1, station2) in a .out record.

    ISBI stores them as s2*(s2+1)/2 + s1 with s1 <= s2, so for three stations:
    0=(0,0) 1=(0,1) 2=(1,1) 3=(0,2) 4=(1,2) 5=(2,2). The two-station layout
    0=auto, 1=cross, 2=auto is just the first three of these.
    """
    s1, s2 = min(station1, station2), max(station1, station2)
    return s2 * (s2 + 1) // 2 + s1


def print_header(header: Header) -> None:
    for field_name in header.__dataclass_fields__:
        print(f"{field_name}: {getattr(header, field_name)}")

def read_subband(output_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    headers = []
    visibilities = []

    with open(output_file_path, "rb") as file:
        while True:
            try:
                magic = struct.unpack("I", file.read(4))[0]
                nr_receivers = struct.unpack("H", file.read(2))[0]
                nr_polarizations = struct.unpack("B", file.read(1))[0]
                correlation_mode = struct.unpack("B", file.read(1))[0]
                start_time = struct.unpack("d", file.read(8))[0]
                end_time = struct.unpack("d", file.read(8))[0]
                # SIGNED: the correlator computes valid_samples/channels - 15
                # and stores it as uint32, so a baseline with (almost) no valid
                # data comes out negative -- read unsigned, -15 becomes
                # 4294967281 and swamps any weighted average it enters. Use
                # utils.process_data.extract_weights, which also clips at 0.
                weights = struct.unpack("i" * 300, file.read(4 * 300))
                nr_samples_per_integration = struct.unpack("I", file.read(4))[0]
                nr_channels = struct.unpack("H", file.read(2))[0]
                file.read(2)
                first_channel_frequency = struct.unpack("d", file.read(8))[0]
                channel_bandwidth = struct.unpack("d", file.read(8))[0]
                file.read(288)

                header = Header(
                    magic=magic,
                    nr_receivers=nr_receivers,
                    nr_polarizations=nr_polarizations,
                    correlation_mode=correlation_mode,
                    start_time=start_time,
                    end_time=end_time,
                    weights=weights,
                    nr_samples_per_integration=nr_samples_per_integration,
                    nr_channels=nr_channels,
                    first_channel_frequency=first_channel_frequency,
                    channel_bandwidth=channel_bandwidth,
                )

                vis_dtype = np.complex64
                nr_baselines = header.nr_receivers * (header.nr_receivers + 1) // 2
                vis_shape = (header.nr_channels, nr_baselines, 2, 2)

                n_items = np.prod(vis_shape)
                n_bytes = n_items * np.dtype(vis_dtype).itemsize

                vis = file.read(n_bytes)
                if len(vis) < n_bytes:
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


def read_all(output_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    all_headers = []
    all_visibilities = []

    for file in output_files:
        headers, visibilities = read_subband(file)
        all_headers.append(headers)
        all_visibilities.append(visibilities)

    return np.asarray(all_headers, dtype=object), np.stack(all_visibilities)


if __name__ == "__main__":
    output_files = [
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_1.out",
            "/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/results/temp/subband_2.out"
    ]

    subband_0_headers, subband_0_data = read_subband(output_files[0])
    print(subband_0_headers.shape)
    print(subband_0_data.shape)

    subband_headers, subband_data = read_all(output_files)
    print(subband_headers.shape)
    print(subband_data.shape)

    print(print_header(subband_0_headers[0]))
