#!/usr/bin/env python3

import numpy as np
import argparse
import struct
import json
import vex
import os

from astropy.time import Time, TimeDelta
from astropy import coordinates as ac
from astropy import units as un
from pycalc11 import Calc

REFERENCE_STATION = "Ib"

def parse_vex_time(time_str):
    year = int(time_str[:4])
    day_of_year = int(time_str[5:8])
    hour = int(time_str[9:11])
    minute = int(time_str[12:14])
    second = int(time_str[15:17])

    date = Time(f"{year}-01-01T00:00:00.000", format="isot", scale="utc") + TimeDelta(day_of_year - 1, format="jd")
    formatted_date = f"{date.strftime('%Y-%m-%d')}T{hour:02}:{minute:02}:{second:02}.000"
    return Time(formatted_date, format="isot", scale="utc")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate delay for ISBI-AARTFAAC correlator'
    )
    parser.add_argument('control', help='Path to control JSON file')

    return parser.parse_args()

def extract_duration(vex_file, scan_nr):
    scan_info = vex_file["SCHED"][scan_nr]
    duration_str = scan_info["station"][2].split()[0]
    return int(duration_str)

def extract_clock_offsets(vex_file):
    clock_block = vex_file['CLOCK']
    clock_offsets = {}
    for x in clock_block:
        clock_offsets[x] = float(clock_block[x].get('clock_early')[1].split()[0]) * 1e-6
    return clock_offsets

def extract_clock_rates(vex_file):
    clock_block = vex_file['CLOCK']
    clock_rates = {}
    for x in clock_block:
        clock_rates[x] = float(clock_block[x].get('clock_early')[3])

    return clock_rates

def extract_center_frequencies(vex_file):
    freq_block = vex_file['FREQ']
    freq_key = list(freq_block.keys())[0]
    freq_block = freq_block[freq_key]
    all_chandefs = freq_block.getall('chan_def')

    center_frequencies = set()
    for chan_def in all_chandefs:
        freq = float(chan_def[1].split()[0])
        bound = chan_def[2]
        bandwidth = float(chan_def[3].split()[0])

        if bound == 'L':
            freq -= bandwidth / 2
        elif bound == 'U':
            freq += bandwidth / 2

        center_frequencies.add(freq)

    return sorted(float(freq) * 1e6 for freq in center_frequencies) # Convert MHz to Hz

def extract_channel_mapping(vex_file):
    threads_block = vex_file['THREADS']
    thread_key = list(threads_block.keys())[0]
    threads_block = threads_block[thread_key]
    all_channels = threads_block.getall('channel')
    channel_mapping = []
    for channel in all_channels:
        channel_nr = int(channel[2])
        channel_mapping.append(channel_nr)
    return channel_mapping

def save_config(path, delays, center_frequencies, channel_mapping):
    with open(path, "wb") as file:
        for delay in delays:
            file.write(struct.pack("i", len(delays[delay])))
            file.write(struct.pack("d" * len(delays[delay]), *delays[delay]))

        file.write(struct.pack("i", len(center_frequencies)))
        file.write(struct.pack("d" * len(center_frequencies), *center_frequencies))
        file.write(struct.pack("i", len(channel_mapping)))
        file.write(struct.pack("i" * len(channel_mapping), *channel_mapping))

def extract_start_time(vex_file, scan_nr):
    scan_info = vex_file["SCHED"][scan_nr]
    start_str = scan_info["start"]
    start_time = parse_vex_time(start_str)
    return start_time

def geometric_delays(vex_file, scan_nr, n_integrations):
    other_stations = ["Ir"]
    station_names = [REFERENCE_STATION] + other_stations

    station_sites = [vex_file["STATION"][station]["SITE"] for station in station_names]

    station_locations = {}
    for station in station_names:
        site = vex_file["STATION"][station]["SITE"]
        positions = vex_file["SITE"][site]["site_position"]
        coords = [float(pos.split()[0]) * un.m for pos in positions]
        station_locations[station] = ac.EarthLocation.from_geocentric(*coords)

    scan_info = vex_file["SCHED"][scan_nr]

    source = vex_file["SOURCE"][scan_info["source"]]
    source_coords = ac.SkyCoord(
        ra=[source["ra"]],
        dec=[source["dec"]],
        frame="fk5",
        equinox="J2000",
    )

    start_time = extract_start_time(vex_file, scan_nr)
    duration_str = scan_info["station"][2].split()[0]
    duration_min = int(duration_str) / 60

    ci = Calc(
        station_names=station_sites,
        station_coords=list(station_locations.values()),
        source_coords=source_coords,
        start_time=start_time,
        duration_min=duration_min,
    )
    ci.run_driver()

    duration_sec = duration_min * 60

    time_offsets = np.linspace(0, duration_sec, n_integrations)
    fine_time_grid = start_time + TimeDelta(time_offsets, format="sec")
    high_res_delays = ci.interpolate_delays(fine_time_grid)

    station_list = list(station_locations.keys())
    g_delays = {station: [] for station in station_list}

    for delay in high_res_delays:
        for i, station in enumerate(station_list):
            g_delays[station].append(delay[0][i][0].value)

    g_delays["Ib"] = np.array(g_delays["Ib"])
    g_delays["Ir"] = np.array(g_delays["Ir"])

    return g_delays, time_offsets

def parse_gps(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # split fields with whitespace
            parts = line.split()
            if len(parts) != 4:
                continue  # skip lines which don't have exactly 4 columns
            mjd, offset, rms, gps_name = parts
            try:
                entry = {
                    'mjd': float(mjd),
                    'offset_us': float(offset),
                    'rms_us': float(rms),
                    'gps_name': gps_name
                }
                data.append(entry)
            except ValueError:
                continue  # skip lines that can't be parsed
    return data

def get_offset_for_mjd(gps_data, target_mjd):
    for entry in gps_data:
        if np.floor(entry['mjd']) == target_mjd:
            return entry['offset_us'] * 1e-6
    return 0.0

if __name__ == '__main__':
    args = parse_arguments()

    with open(args.control) as f:
        ctrl_file = json.load(f)

    with open(ctrl_file['vex-path']) as f:
        vex_file = vex.parse(f.read())

    scan_nr = ctrl_file['scan-number']
    duration = extract_duration(vex_file, scan_nr)
    n_integrations = int(duration / ctrl_file['integration-time']) + 1
    g_delays, time_offsets = geometric_delays(vex_file, scan_nr, n_integrations=n_integrations)
    clock_offsets = extract_clock_offsets(vex_file)
    clock_rates = extract_clock_rates(vex_file)
    center_frequencies = extract_center_frequencies(vex_file)
    channel_mapping = extract_channel_mapping(vex_file)
    subbands = ctrl_file['subbands']
    start_time = extract_start_time(vex_file, scan_nr)
    start_time_mjd = np.floor(start_time.mjd)

    gps_ib = parse_gps('./gps(1).ib')
    gps_ir = parse_gps('./gps(1).ir')

    offset_ib = get_offset_for_mjd(gps_ib, start_time_mjd)
    offset_ir = get_offset_for_mjd(gps_ir, start_time_mjd)

    g_delays['Ir'] -= offset_ir
    g_delays['Ib'] -= offset_ib

    # right now I'm adding some extra delay that I calculate from the lags
    # TODO: investigate what else can be done, cus this is probably not good
    # g_delays['Ir'] -= (2.0e-6 + 7.65625e-7 + 6.25e-8 / 4 - 2.5e-7)
    g_delays['Ir'] -= (2.5e-6)

    for d in g_delays:
        g_delays[d] = -g_delays[d]

    # subband 1 = index 0 and 1, subband 2 = index 2 and 3, ...
    selected_indices = []
    for subband in subbands:
        selected_indices.extend([2 * (subband - 1), 2 * (subband - 1) + 1])

    center_frequencies = [center_frequencies[i - 1] for i in subbands]
    channel_mapping = [channel_mapping[i] for i in selected_indices]

    output_path = f'{ctrl_file["output-path"]}/{ctrl_file["experiment"]}/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_config(
        output_path + f"{scan_nr}.conf",
        g_delays,
        center_frequencies,
        channel_mapping
    )
