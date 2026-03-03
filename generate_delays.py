#!/usr/bin/env python3
"""Generate geometric delay tables for the ISBI-AARTFAAC correlator.

Extracts station, source, and scheduling information from a VEX observation
file and computes geometric delays using pycalc11. The resulting delays,
center frequencies, and channel mappings are written to a binary configuration
file (.conf) consumed by the correlator.
"""

import numpy as np
import argparse
import struct
import json
import os

from astropy.time import TimeDelta
from pycalc11 import Calc

from vextractor import VEXtractor


def save_config(path, delays, center_frequencies, channel_mapping):
    """Write the correlator configuration to a binary file.

    File format (all little-endian):
        For each station: int32 n_delays, float64[n_delays] delay values
        int32 n_frequencies, float64[n_frequencies] center frequencies in Hz
        int32 n_channels, int32[n_channels] channel mapping indices
    """
    with open(path, "wb") as file:
        for station, values in delays.items():
            file.write(struct.pack("i", len(values)))
            file.write(struct.pack("d" * len(values), *values))

        file.write(struct.pack("i", len(center_frequencies)))
        file.write(struct.pack("d" * len(center_frequencies), *center_frequencies))
        file.write(struct.pack("i", len(channel_mapping)))
        file.write(struct.pack("i" * len(channel_mapping), *channel_mapping))


def geometric_delays(vextractor, scan_nr, n_integrations, reference_station="Ib"):
    """Compute geometric delays for all stations using pycalc11.

    Args:
        vextractor: VEXtractor instance for the observation.
        scan_nr: Scan identifier (e.g. 'No0002').
        n_integrations: Number of evenly-spaced time points to evaluate.
        reference_station: Station to place first in the delay model.

    Returns:
        Tuple of (g_delays, time_offsets) where g_delays is a dict mapping
        station names to numpy arrays of delay values in seconds, and
        time_offsets is the array of time offsets in seconds from scan start.
    """
    all_stations = vextractor.stations()
    station_names = [reference_station] + [s for s in all_stations if s != reference_station]

    station_sites, station_locations = vextractor.station_locations(station_names)
    source_coords = vextractor.source_coords(scan_nr)

    start_time = vextractor.start_time(scan_nr)
    duration_sec = vextractor.duration(scan_nr)
    duration_min = duration_sec / 60

    ci = Calc(
        station_names=station_sites,
        station_coords=list(station_locations.values()),
        source_coords=source_coords,
        start_time=start_time,
        duration_min=duration_min,
    )
    ci.run_driver()

    time_offsets = np.linspace(0, duration_sec, n_integrations)
    fine_time_grid = start_time + TimeDelta(time_offsets, format="sec")
    high_res_delays = ci.interpolate_delays(fine_time_grid)

    station_list = list(station_locations.keys())
    g_delays = {station: [] for station in station_list}

    for delay in high_res_delays:
        for i, station in enumerate(station_list):
            g_delays[station].append(delay[0][i][0].value)

    for station in station_list:
        g_delays[station] = np.array(g_delays[station])

    return g_delays, time_offsets
