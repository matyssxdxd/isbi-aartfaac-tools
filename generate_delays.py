#!/usr/bin/env python3
import argparse
import json
import os
import struct
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import vex
from astropy import coordinates as ac
from astropy import units as un
from astropy.time import Time, TimeDelta
from pycalc11 import Calc

REFERENCE_STATION = "Ib"

def parse_vex_time(time_str):
    year = int(time_str[:4])
    day_of_year = int(time_str[5:8])
    hour = int(time_str[9:11])
    minute = int(time_str[12:14])
    second = int(time_str[15:17])

    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    formatted_date = f"{date.strftime('%Y-%m-%d')}T{hour:02}:{minute:02}:{second:02}.000"
    return Time(formatted_date, format="isot", scale="utc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate delay for ISBI-AARTFAAC correlator"
    )
    parser.add_argument("vex", help="Path to VEX observation file")
    parser.add_argument("scan", help="Scan identifier from VEX SCHED section")
    # parser.add_argument("lag_file", help="File with lag-fit delays (optional)")
    args = parser.parse_args()

    with open(args.vex) as f:
        v = vex.parse(f.read())

    scan = args.scan

    other_stations = ["Ir"]
    station_names = [REFERENCE_STATION] + other_stations

    station_sites = [v["STATION"][station]["SITE"] for station in station_names]

    station_locations = {}
    for station in station_names:
        site = v["STATION"][station]["SITE"]
        positions = v["SITE"][site]["site_position"]
        coords = [float(pos.split()[0]) * un.m for pos in positions]
        station_locations[station] = ac.EarthLocation.from_geocentric(*coords)

    scan_info = v["SCHED"][scan]

    source = v["SOURCE"][scan_info["source"]]
    source_coords = ac.SkyCoord(
        ra=[source["ra"]],
        dec=[source["dec"]],
        frame="fk5",
        equinox="J2000",
    )

    start_str = scan_info["start"]
    start_time = parse_vex_time(start_str)
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

    N = 181
    duration_sec = duration_min * 60

    time_offsets = np.linspace(0, duration_sec, N)
    fine_time_grid = start_time + TimeDelta(time_offsets, format="sec")
    high_res_delays = ci.interpolate_delays(fine_time_grid)

    station_list = list(station_locations.keys())
    g_delays = {station: [] for station in station_list}

    for delay in high_res_delays:
        for i, station in enumerate(station_list):
            g_delays[station].append(delay[0][i][0].value)

    g_delays["Ib"] = np.array(g_delays["Ib"])
    g_delays["Ir"] = np.array(g_delays["Ir"])
    
    # print(g_delays["Ib"])
    # print(g_delays["Ir"])
    
    
    # rate_ir = 2.02e-07
    # rate_ib = 1.99e-07
    # offset_ir = 8.233828125e-6
    # offset_ib = 8.114e-6
    # total = offset_ir - offset_ib
    # total_rate = rate_ir - rate_ib
    
    # rates = []
    # for i in range(1, N + 1):
    #     rates.append(total_rate * i)
        
    # rates = np.array(rates)
    # g_delays["Ir"] += total
    # g_delays["Ir"] += rates
    
        # Use baseline Ib-Ir delay model from visibility fit instead:
    delay_rate_avg =-3.110075e-09
    clock_offset_avg =-5.615988e-07
    # time_offsets already runs from 0 .. duration_sec with N points
    baseline_correction = clock_offset_avg + delay_rate_avg * time_offsets  # shape (N,)

    # Apply to Ir wrt Ib
    g_delays["Ir"] += baseline_correction
    
    center_frequencies = [6667.69]
    channel_mapping = [9, 13]
    config_path = "./No0002.conf"
    import struct

    with open(config_path, "wb") as file:
        for delay in g_delays:
            print(delay)
            file.write(struct.pack("i", len(g_delays[delay])))
            file.write(struct.pack("d" * len(g_delays[delay]), *g_delays[delay]))

        file.write(struct.pack("i", len(center_frequencies)))
        file.write(struct.pack("d" * len(center_frequencies), *center_frequencies))
        file.write(struct.pack("i", len(channel_mapping)))
        file.write(struct.pack("i" * len(channel_mapping), *channel_mapping))



    
