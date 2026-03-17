import numpy as np
import astropy.units as u

from pycalc11 import Calc
from vextractor import VEXtractor
from astropy.time import TimeDelta

def pycalc11_delays(vex, scan, reference_station="Ib"):
    all_stations = vex.stations()
    station_names = [reference_station] + [s for s in all_stations if s != reference_station]

    station_sites, station_locations = vex.station_locations(station_names)
    source_coords = vex.source_coords(scan)

    start_time = vex.start_time(scan)
    duration_sec = vex.duration(scan)
    duration_min = duration_sec // 60

    ci = Calc(
        station_names=station_sites,
        station_coords=list(station_locations.values()),
        source_coords=source_coords,
        start_time=start_time,
        duration_min=duration_min
    )
    ci.run_driver()

    time_offsets = np.arange(-1, duration_sec + 1, 1) # padding of 1 second on each side
    fine_time_grid = start_time + TimeDelta(time_offsets, format="sec")
    high_res_delays = ci.interpolate_delays(fine_time_grid)

    station_list = list(station_locations.keys())
    g_delays = {station: [] for station in station_list}

    for delay in high_res_delays:
        for i, station in enumerate(station_list):
            g_delays[station].append(delay[0][i][0].value)

    start_time_unix = int(start_time.unix)
    sample_rate = int(vex.sample_rate())
    scan_start_samples = start_time_unix * sample_rate

    print(scan_start_samples)

    final = {}

    x = scan_start_samples + np.rint(time_offsets * sample_rate).astype(np.int64)

    clock_rates = vex.clock_rates()
    clock_offsets = vex.clock_offsets()
    t_abs = start_time + time_offsets * u.s
    for station, d in g_delays.items():
        ce = vex.clock_epoch()[station]
        sec_clock = (t_abs - ce).to_value(u.s)
        d = np.asarray(d, np.float64)
        g_delays[station] = d + clock_offsets[station] + sec_clock * clock_rates[station]

    for station, d in g_delays.items():
        arr = np.empty(len(x), dtype=[("timestamp", np.int64), ("delay", np.float64)])
        arr["timestamp"] = x
        arr["delay"] = d
        final[station] = arr

    return final

if __name__ == '__main__':
    vex = VEXtractor('./E011/E011.vix')
    scan = 'No0002'
    pycalc11_delays(vex, scan)

