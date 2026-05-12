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

# TODO: Pass vex path instead of VEXtractor object?
def sfxc_delays(vex, delay_paths, scan_name, reference_station):
    """
    Read SFXC delay files for a scan and return clock-corrected delays by station.

    The returned delays are converted to sample-based timstamps relative to the
    scan start time and corrected using station clock offsets and clock rates
    from the VEX file. The reference station is placed first in the returned
    dictionary.

    Args:
        vex: VEXTractor object.
        delay_paths: Mapping from station name to SFXC `.del` file path.
        scan_name: Name of the scan (usually in Noxxxx format) whose delay should be extracted.
        reference_station: Station name to place first in the returned dictionary.

    Returns:
        Mapping from station name to a structured NumPy array with fields:
            - timestamp: Sample timestamp for each delay point.
            - delay: Clock-corrected delay value in seconds.

    Raises:
        KeyError: If `reference_station` is not present in `delay_paths`
            or no delays were loaded for it.
        ValueError: If `scan_name` is not found in one of the delay files.
    """
    delays = {}
    sod_by_station = {}

    sample_rate = vex.sample_rate()
    scan_start = vex.start_time(scan_name)

    scan_start_samples = int(np.round(scan_start.unix * sample_rate))
    scan_start_sod = (
        scan_start.datetime.hour * 3600
        + scan_start.datetime.minute * 60
        + scan_start.datetime.second
        + scan_start.datetime.microsecond / 1e6
    )

    for station, delay_file in delay_paths.items():
        sod, delay = read_delays(delay_file, scan_name)
        sod = np.asarray(sod, dtype=np.float64)
        delay = np.asarray(delay, dtype=np.float64)

        sod_by_station[station] = sod

        delta_sec = sod - scan_start_sod
        timestamp = (
            scan_start_samples + np.rint(delta_sec * sample_rate)
        ).astype(np.int64)

        arr = np.empty(
            len(delay),
            dtype=[("timestamp", np.int64), ("delay", np.float64)],
        )

        arr["timestamp"] = timestamp
        arr["delay"] = delay

        delays[station] = arr

    clock_offsets = vex.clock_offsets()
    clock_rates = vex.clock_rates()
    clock_epochs = vex.clock_epoch()

    for station, arr in delays.items():
        sod = sod_by_station[station]

        # Calculates time offset (in seconds) relative to the scan start
        delta_sec = sod - scan_start_sod

        # Convert time offsets back to absolute timestamps
        t_abs = scan_start + delta_sec * u.s

        # Convert absolute timestamps to seconds since the station reference
        # clock epoch
        sec_clock = (t_abs - clock_epochs[station]).to_value(u.s)

        arr["delay"] += (
            clock_offsets[station] + sec_clock * clock_rates[station]
        )

    if reference_station not in delays:
        raise KeyError(f"Reference station '{reference_station}' not found in delays")

    # TODO: Instead of this, could save some station 'key' like Ib, Ir
    # Currently it makes sure that the delays for reference station come first
    ordered = {reference_station: delays[reference_station]}
    for station, arr in delays.items():
        if station != reference_station:
            ordered[station] = arr

    return ordered
