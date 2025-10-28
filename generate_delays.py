import numpy as np
import argparse
import struct
import json
import vex
import sys
import os

from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from astropy import coordinates as ac
from astropy import units as un
from astropy.time import Time
from pycalc11 import Calc

parser = argparse.ArgumentParser()
parser.add_argument('vex')
parser.add_argument('control')

args = parser.parse_args()

with open(args.control, 'r') as ctrl_f:
    c = json.load(ctrl_f)

with open(args.vex, 'r') as vex_f:
    v = vex.parse(vex_f.read())

reference_station = c['reference-station']
stations = {station: v['STATION'][station]['SITE'] for station in [reference_station] + [station for station in c['stations'] if station != reference_station]}
subbands = c['subbands']
n_subbands = len(subbands)

station_pos = {station: [float(pos.split()[0]) * un.m
    for pos in v['SITE'][stations[station]]['site_position']]
    for station in stations}

station_loc = {station: ac.EarthLocation.from_geocentric(
    station_pos[station][0],
    station_pos[station][1],
    station_pos[station][2])
    for station in stations}

scans = c['scans']

# TODO: Duration should depend on the one in ctrl file not in VEX file?
for scan in scans:
    source_name = v['SCHED'][scan]['source']
    source_ra = v['SOURCE'][source_name]['ra']
    source_dec = v['SOURCE'][source_name]['dec']
    source_coords = ac.SkyCoord([source_ra], [source_dec], frame='fk5', equinox='J2000.0')
    start_time = v['SCHED'][scan]['start']
    date = datetime(int(start_time[:4]), 1, 1) + timedelta(days=int(start_time[5:8]) - 1)
    formatted_date = date.strftime(f"%Y-%m-%dT{int(start_time[9:11]):02}:{int(start_time[12:14]):02}:{int(start_time[15:17]):02}.000")
    start_time = Time(formatted_date, format='isot', scale='utc')
    # TODO:
    duration = int(v['SCHED'][scan]['station'][2].split()[0]) / 60

    print([stations[k] for k in stations])
    print([station_loc[k] for k in station_loc])
    ci = Calc(
        station_names=[stations[k] for k in stations],
        station_coords=[station_loc[k] for k in station_loc],
        source_coords=source_coords,
        start_time=start_time,
        duration_min=duration
    )

    ci.run_driver()

    s = [station for station in stations]
    delays = {station: np.array([], dtype=np.float64) for station in s}
    delays_intepolated = {station: np.array([], dtype=np.float64) for station in s}
    n_delays = len(ci.delay)

    for delay in ci.delay:
        for i in range(len(delay[0])):
            delays[s[i]] = np.append(delays[s[i]], delay[0][i][0].value)

    x = np.linspace(0, 1, n_delays, dtype=np.float64)
    # TODO:
    duration_seconds = int(duration * 60)
    x_n = np.linspace(0, 1, duration_seconds // int(c['integration_time']) + 1, dtype=np.float64)

    for station in delays:
        interp = CubicSpline(x, delays[station])
        delays_intepolated[station] = interp(x_n)

    center_frequencies = np.array([], dtype=np.float64)

    freq_key = list(v['FREQ'].keys())[0]
    chan_def = v['FREQ'][freq_key].getall('chan_def')

    for i in range(0, len(chan_def), 2):
        current_frequency = float(chan_def[i][1].split()[0])
        subband_bandwidth = float(chan_def[i][3].split()[0])
        bound = chan_def[i][2]
        if bound == 'L':
            center_frequency = current_frequency - subband_bandwidth / 2
        else:
            center_frequency = current_frequency + subband_bandwidth / 2
        center_frequencies = np.append(center_frequencies, center_frequency)

    center_freqs = []

    for subband in subbands:
        center_freqs.append(center_frequencies[subband - 1])

    # TODO: Right now I assume that there is a $THREADS block, but it should be generated if there is not one

    threads_key = list(v['THREADS'].keys())[0]
    threads = v['THREADS'][threads_key].getall('channel')
    mapping = np.zeros(len(threads), dtype=int)

    for i in range(len(threads)):
        thread = threads[i]
        mapping[i] = int(thread[-1])

    channel_mapping = []

    for subband in subbands:
        end_idx = subband * 2 - 1
        start_idx = end_idx - 1
        channel_mapping.extend(mapping[start_idx:end_idx+1])

    output_path = f'{c["output-path"]}/{c["exper_name"]}/{scan}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/{scan}.delays', 'wb') as file:
        for delay in delays_intepolated:
            file.write(struct.pack('i', len(delays_intepolated[delay])))
            file.write(struct.pack('d' * len(delays_intepolated[delay]), *delays_intepolated[delay]))

        file.write(struct.pack('i', len(center_freqs)))
        file.write(struct.pack('d' * len(center_freqs), *center_freqs))

        file.write(struct.pack('i', len(channel_mapping)))
        file.write(struct.pack('i' * len(channel_mapping), *channel_mapping))

