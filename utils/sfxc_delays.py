import numpy as np
import astropy.units as u

from utils.delay_file_reader import DelayFileReader

def read_delays(file, scan_name):
    """
    Read delay data for a specific scan from SFXC `.del` file.

    Args:
        file: Path to the `.del` file to read.
        scan_name: Name of the scan (usually in Noxxxx format) whose delay should be extracted.

    Returns:
        A tuple containing:
            - sec_of_day: List of seconds-of-day values for each data point in the scan.
            - delays: List of delay values corresponding to each data point.

    Raises:
        ValueError: If `scan_name` is not found in one of the delay files.
    """
    reader = DelayFileReader(file)
    reader.read_file()

    matching_scans = [scan for scan in reader.scans if scan["scan_name"] == scan_name]

    if not matching_scans:
        raise ValueError(f"Scan '{scan_name}' not found in {file}")

    scan = matching_scans[0]
    sec_of_day = []
    delays = []

    for point in scan["points"]:
        sec_of_day.append(point["sec_of_day"])
        delays.append(point["delay"])

    return sec_of_day, delays

def sfxc_delays(vex, delay_paths, scan_name, stations, sample_rate, clock_offsets_usec=None):
    """
    Read SFXC delay files for a scan and return clock-corrected delays by station.

    SFXC's .del files hold geometry only; the station clocks are added here from
    the vex $CLOCK entry IN FORCE AT THIS SCAN (a vex can carry several
    clock_early entries per station, each valid from its own start time).

    Args:
        vex: VEXtractor object.
        delay_paths: Mapping from station name to SFXC `.del` file path.
        scan_name: Name of the scan (usually in Noxxxx format) whose delay should be extracted.
        stations: Stations to return, IN ISBI's -i ORDER, reference first. ISBI
            reads exactly one table per -n station from --delayFile, in order.
        sample_rate: The correlator's -f in Hz -- the unit of the table
            timestamps. In a mixed-rate run this is the LOWEST station rate.
        clock_offsets_usec: Optional {station: usec} added on top of the vex clock.

    Returns:
        Mapping from station name, ordered as `stations`, to a structured NumPy
        array with fields:
            - timestamp: Sample timestamp (unix seconds x sample_rate).
            - delay: Clock-corrected delay value in seconds.

    Raises:
        KeyError: If a station has no entry in `delay_paths`.
        ValueError: If `scan_name` is not found in one of the delay files.
    """
    clock_offsets_usec = clock_offsets_usec or {}
    scan_start = vex.start_time(scan_name)

    scan_start_samples = int(np.round(scan_start.unix * sample_rate))
    scan_start_sod = (
        scan_start.datetime.hour * 3600
        + scan_start.datetime.minute * 60
        + scan_start.datetime.second
        + scan_start.datetime.microsecond / 1e6
    )

    clock_offsets = vex.clock_offsets(scan_start)
    clock_rates = vex.clock_rates(scan_start)
    clock_epochs = vex.clock_epoch(scan_start)

    delays = {}
    for station in stations:
        if station not in delay_paths:
            raise KeyError(f"No SFXC delay file given for station '{station}'")

        sod, delay = read_delays(delay_paths[station], scan_name)
        sod = np.asarray(sod, dtype=np.float64)
        delay = np.asarray(delay, dtype=np.float64)

        # Calculates time offset (in seconds) relative to the scan start
        delta_sec = sod - scan_start_sod
        timestamp = (scan_start_samples + np.rint(delta_sec * sample_rate)).astype(np.int64)

        # Seconds since the station's clock-rate reference epoch
        t_abs = scan_start + delta_sec * u.s
        epoch = clock_epochs[station]
        sec_clock = (t_abs - epoch).to_value(u.s) if epoch is not None else 0.0

        arr = np.empty(len(delay), dtype=[("timestamp", np.int64), ("delay", np.float64)])
        arr["timestamp"] = timestamp
        # Kept as delay + (offset + rate term), the association the earlier
        # version used, so .conf files regenerate bit-identically.
        arr["delay"] = delay + (clock_offsets[station] + sec_clock * clock_rates[station])
        if station in clock_offsets_usec:
            arr["delay"] += clock_offsets_usec[station] * 1e-6
        delays[station] = arr

    return delays
