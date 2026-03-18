import struct
import argparse
import numpy as np

def parse_arguments(description, args):
    parser = argparse.ArgumentParser(
        description=description
    )

    for arg, help in args.items():
        parser.add_argument(arg, help=help)

    return parser.parse_args()

def sod_to_hms(sod):
    sod = float(sod)

    h = int(sod // 3600)
    m = int((sod % 3600) // 60)
    s = sod % 60  # keep fractional part

    return f"{h:02d}:{m:02d}:{s:06.3f}"  # 2 digits + .mmm

def save_config(path, delays, center_frequencies, channel_mapping):
    with open(path, "wb") as file:
        for station, values in delays.items():
            file.write(struct.pack("<i", len(values)))
            file.write(values.tobytes())

        center_frequencies = np.asarray(center_frequencies, dtype="<f8")
        channel_mapping = np.asarray(channel_mapping, dtype="<i4")

        file.write(struct.pack("<i", len(center_frequencies)))
        file.write(center_frequencies.tobytes())

        file.write(struct.pack("<i", len(channel_mapping)))
        file.write(channel_mapping.tobytes())
