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
    s = sod % 60

    return f"{h:02d}:{m:02d}:{s:06.3f}"

def save_config(path, delays, stations):
    """Write ISBI's --delayFile.

    Format, per station in order: uint32 count, then count x (int64 timestamp,
    float64 delay seconds), little-endian; the timestamp is unix seconds x the
    correlator's -f. ISBI reads exactly one table per -n station, in -i order,
    and linearly interpolates in each, so:
      - `stations` must be the -i list, reference first (ISBI subtracts the
        FIRST table from all the others);
      - each table must be sorted and strictly increasing in time.

    Frequencies and the channel mapping used to be appended here too; ISBI no
    longer reads them from this file, they are command-line options now.
    """
    with open(path, "wb") as file:
        for station in stations:
            values = np.ascontiguousarray(
                delays[station].astype([("timestamp", "<i8"), ("delay", "<f8")]))
            if len(values) < 2:
                raise ValueError(f"Delay table for '{station}' has {len(values)} entries; "
                                 "ISBI interpolates and needs at least 2")
            if np.any(np.diff(values["timestamp"]) <= 0):
                raise ValueError(f"Delay table for '{station}' is not strictly increasing in time")
            file.write(struct.pack("<I", len(values)))
            file.write(values.tobytes())
