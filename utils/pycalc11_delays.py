import numpy as np
import astropy.units as u

from pycalc11 import Calc
from utils.vextractor import VEXtractor
from astropy.time import TimeDelta


def pycalc11_delays(vex, scan, stations, sample_rate, clock_offsets_usec=None, padding_sec=2):
    """Geometric + clock delays from CALC (pycalc11) for one scan.

    Args:
        vex: VEXtractor.
        scan: Scan name, e.g. 'No0006'.
        stations: Stations to compute, IN ISBI's -i ORDER, reference first.
            ISBI reads exactly one table per -n station from its --delayFile, in
            that order, so this list and the -i list must be the same list.
            (This used to compute every station in the vex, in vex order -- with
            a 16-station vex the tables ISBI read did not belong to its inputs.)
        sample_rate: The correlator's -f, in Hz. Table timestamps are in units
            of 1/-f, so in a mixed-rate run this is the LOWEST station rate, not
            any one station's own rate.
        clock_offsets_usec: Optional {station: usec} added on top of the vex
            clock. For stations whose vex has no $CLOCK (n26l2) this is the
            whole clock, found by a fringe search.
        padding_sec: Seconds of table before and after the scan. ISBI throws if
            it needs a delay outside the table, and it reads ahead of the start
            for filter history and the maximum delay.

    Returns:
        {station: structured array (timestamp int64, delay float64)}, ordered as
        `stations`.
    """
    clock_offsets_usec = clock_offsets_usec or {}
    station_sites, station_locations = vex.station_locations(stations)
    source_coords = vex.source_coords(scan)

    start_time = vex.start_time(scan)
    duration_sec = vex.duration(scan)

    # pycalc11 works in whole minutes; cover the scan plus padding on both
    # sides. Integer division used to give 0 minutes for any scan under 60 s.
    calc_start = start_time - TimeDelta(60, format="sec")
    duration_min = int(np.ceil((duration_sec + 2 * padding_sec) / 60.0)) + 2

    ci = Calc(
        station_names=station_sites,
        station_coords=list(station_locations.values()),
        source_coords=source_coords,
        start_time=calc_start,
        duration_min=duration_min,
    )
    ci.run_driver()

    time_offsets = np.arange(-padding_sec, duration_sec + padding_sec + 1, 1)
    fine_time_grid = start_time + TimeDelta(time_offsets, format="sec")
    high_res_delays = ci.interpolate_delays(fine_time_grid)

    station_list = list(station_locations.keys())
    g_delays = {station: [] for station in station_list}

    for delay in high_res_delays:
        for i, station in enumerate(station_list):
            g_delays[station].append(delay[0][i][0].value)

    # Exact: start_time is on an integer second, and -f is an integer.
    scan_start_samples = int(round(start_time.unix)) * int(sample_rate)
    x = scan_start_samples + (time_offsets * int(sample_rate)).astype(np.int64)

    # The clock in force AT THIS SCAN, not simply the first clock_early entry.
    clock_rates = vex.clock_rates(start_time)
    clock_offsets = vex.clock_offsets(start_time)
    clock_epochs = vex.clock_epoch(start_time)
    t_abs = start_time + time_offsets * u.s

    final = {}
    for station in stations:
        d = np.asarray(g_delays[station], np.float64)
        epoch = clock_epochs[station]
        sec_clock = (t_abs - epoch).to_value(u.s) if epoch is not None else 0.0
        d = (d + clock_offsets[station] + sec_clock * clock_rates[station]
             + clock_offsets_usec.get(station, 0.0) * 1e-6)

        arr = np.empty(len(x), dtype=[("timestamp", np.int64), ("delay", np.float64)])
        arr["timestamp"] = x
        arr["delay"] = d
        final[station] = arr

    return final


if __name__ == "__main__":
    vex = VEXtractor("./E011/E011.vix")
    scan = "No0002"
    print(pycalc11_delays(vex, scan, ["Ib", "Ir"], vex.sample_rate()))
