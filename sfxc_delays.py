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
    time_offsets = np.arange(0, n_integrations) * integration_time

    clock_offsets = vex.clock_offsets()
    clock_rates = vex.clock_rates()
    # TODO: Redo the method, because in this case the clock epoch is the same for both stations.
    #       Could it be that the clock epoch is different?
    clock_epoch = vex.clock_epoch()['Ir']
    scan_start = vex.start_time(scan)

    epoch_offset = (clock_epoch - scan_start).to_value('sec')
    delays = {}

    # Read per-station delay tables
    for station, delay_file in delay_paths.items():
        sod, delay = read_delays(delay_file, scan)

        delays[station] = {
            'sod': sod,
            'del': delay,
        }

    # Target times in SOD (sec_of_day) for interpolation
    scan_start_sod = float((scan_start.mjd % 1) * 86400.0)
    target_sod = scan_start_sod + time_offsets

    # Interpolate per station
    for station, d in delays.items():
        sod = d['sod']
        delay = d['del']
        d['interp'] = Akima1DInterpolator(sod, delay, extrapolate=True)(target_sod)

    # Absolute times for each integration (astropy Time array)
    t_abs = scan_start + time_offsets * u.s

    for station, d in delays.items():
        # per-station clock epoch (even if they happen to be equal)
        ce = vex.clock_epoch()[station]
        # seconds since that station's clock epoch
        sec_clock = (t_abs - ce).to_value(u.s)  # float array
        # add clock drift in seconds
        d['interp'] = d['interp'] + clock_offsets[station] + sec_clock * clock_rates[station]

    ordered = {}

    if reference_station not in delays:
        raise KeyError(f"Reference station '{reference_station}' not found in delays")

    # Insert reference station first
    ordered[reference_station] = delays[reference_station]['interp']

    # Insert remaining stations
    for station, d in delays.items():
        if station != reference_station:
            ordered[station] = d['interp']

    return ordered
