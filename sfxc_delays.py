import numpy as np
import os
from vextractor import VEXtractor, parse_vex_time
from generate_delays import geometric_delays, save_config
from utils.delay_file_reader import DelayFileReader
from utils.helpers import parse_arguments
from scipy.interpolate import Akima1DInterpolator
import astropy.units as u


def read_delays(file, scan_name):
    reader = DelayFileReader(file)
    reader.read_file()

    matching_scans = [scan for scan in reader.scans if scan['scan_name'] == scan_name]
    if not matching_scans:
        raise ValueError(f"Scan '{scan_name}' not found in {file}")

    scan = matching_scans[0]
    sec_of_day = []
    delays = []
    for point in scan['points']:
        sec_of_day.append(point['sec_of_day'])
        delays.append(point['delay'])

    return np.array(sec_of_day), np.array(delays)


def sfxc_delays(vex, delay_paths, scan, n_integrations, integration_time, reference_station):
    duration = vex.duration(scan)

    clock_offsets = vex.clock_offsets()
    clock_rates = vex.clock_rates()
    scan_start = vex.start_time(scan)
    scan_start_unix = int(scan_start.unix)
    sample_rate = int(vex.sample_rate())
    scan_start_samples = scan_start_unix * sample_rate
    times_per_block = int(sample_rate * integration_time)

    delays = {}

    # Read per-station delay tables
    for station, delay_file in delay_paths.items():
        sod, delay = read_delays(delay_file, scan)
        delays[station] = {
            "sod": sod,
            "del": delay,
        }

    time_offsets = np.arange(-1, len(delays['Ib']['sod']) - 1, 1) # -1 and +1 beacuse SFXC delays have a padding of 1 second

    x = scan_start_samples + np.rint(time_offsets * sample_rate).astype(np.int64)

    # Apply clock model per station
    t_abs = scan_start + time_offsets * u.s
    print(t_abs)
    for station, d in delays.items():
        ce = vex.clock_epoch()[station]
        sec_clock = (t_abs - ce).to_value(u.s)
        d["del"] = d["del"] + clock_offsets[station] + sec_clock * clock_rates[station]

    final = {}

    for station, d in delays.items():
        arr = np.empty(len(x), dtype=[("timestamp", np.int64), ("delay", np.float64)])
        arr["timestamp"] = x
        arr["delay"] = d["del"]
        final[station] = arr

    if reference_station not in final:
        raise KeyError(f"Reference station '{reference_station}' not found in delays")

    ordered = {reference_station: final[reference_station]}
    for station, arr in final.items():
        if station != reference_station:
            ordered[station] = arr

    return ordered