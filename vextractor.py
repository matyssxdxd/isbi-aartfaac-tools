"""VEXtractor: extract observation parameters from a parsed VEX file.

Provides methods to read station, frequency, clock, and scheduling
information from a VEX observation file.
"""

import vex
import numpy as np

from astropy.time import Time, TimeDelta
from astropy import coordinates as ac
from astropy import units as un


def parse_vex_time(time_str):
    """Parse a VEX timestamp (e.g. '2024y289d13h39m17s') into an astropy Time object."""
    year = int(time_str[:4])
    day_of_year = int(time_str[5:8])
    hour = int(time_str[9:11])
    minute = int(time_str[12:14])
    second = int(time_str[15:17])

    date = Time(f"{year}-01-01T00:00:00.000", format="isot", scale="utc") + TimeDelta(day_of_year - 1, format="jd")
    formatted_date = f"{date.strftime('%Y-%m-%d')}T{hour:02}:{minute:02}:{second:02}.000"
    return Time(formatted_date, format="isot", scale="utc")


class VEXtractor:
    """Extract observation parameters from a parsed VEX file.

    Args:
        vex_path: Path to the VEX file on disk.
    """

    def __init__(self, vex_path):
        with open(vex_path) as f:
            self.vex = vex.parse(f.read())

        freq_block = self.vex['FREQ']
        freq_key = list(freq_block.keys())[0]
        self.freq_def = freq_block[freq_key]

    @staticmethod
    def _as_time(value):
        if isinstance(value, Time):
            return value
        return Time(value)

    @staticmethod
    def _parse_seconds(value):
        """Parse a VEX amount into seconds.

        Accepts strings like '8.233 usec', '0.002 sec', or a bare number.
        """
        tokens = str(value).split()
        number = float(tokens[0])
        if len(tokens) == 1:
            # Bare numbers are interpreted as seconds.
            return number

        unit = tokens[1].lower()
        if unit in ("sec", "s", "second", "seconds"):
            return number
        if unit in ("msec", "ms"):
            return number * 1e-3
        if unit in ("usec", "us"):
            return number * 1e-6
        if unit in ("nsec", "ns"):
            return number * 1e-9

        raise ValueError(f"Unsupported time unit in VEX amount: {value}")

    @classmethod
    def _parse_rate_sec_per_sec(cls, value):
        """Parse VEX clock rate with SFXC-compatible fallback.

        SFXC logic:
        - if units are present: parse as sec/sec amount
        - if units are absent: historically interpret as usec/sec (x1e-6)
        """
        text = str(value).strip()
        tokens = text.split()

        if len(tokens) == 1:
            return float(tokens[0]) * 1e-6

        number = float(tokens[0])
        unit = tokens[1].lower()
        if unit in ("sec/sec", "s/s"):
            return number
        if unit in ("usec/sec", "us/sec", "microsec/sec"):
            return number * 1e-6
        if unit in ("nsec/sec", "ns/sec"):
            return number * 1e-9

        raise ValueError(f"Unsupported clock rate unit in VEX amount: {value}")

    def _clock_entries_for_station(self, station):
        """Return parsed clock_early entries for a station.

        Each entry is a dict with keys: start, offset_sec, rate_sec_per_sec, epoch.
        """
        if station not in self.vex["STATION"]:
            raise KeyError(f"Unknown station '{station}'")

        clock_ref = self.vex["STATION"][station]["CLOCK"]
        clock_block = self.vex["CLOCK"][clock_ref]

        entries = clock_block.getall("clock_early")

        if not entries:
            raise ValueError(f"No clock_early entries for station '{station}' (CLOCK='{clock_ref}')")
        
        parsed = []
        for row in entries:
            start = parse_vex_time(row[0])
            offset_sec = self._parse_seconds(row[1])

            rate_sec_per_sec = 0.0
            epoch = start
            if len(row) > 3:
                epoch = parse_vex_time(row[2])
                rate_sec_per_sec = self._parse_rate_sec_per_sec(row[3])

            parsed.append(
                {
                    "start": start,
                    "offset_sec": offset_sec,
                    "rate_sec_per_sec": rate_sec_per_sec,
                    "epoch": epoch,
                }
            )

        parsed.sort(key=lambda x: x["start"].mjd)
        return parsed

    def clock_entry(self, station):
        """Return the active clock_early entry for a station."""
        entry= self._clock_entries_for_station(station)

        if entry is None:
            raise ValueError(
                f"No clock_early entry for station '{station}'")

        return entry

    def duration(self, scan_nr):
        """Extract the scan duration in seconds from the VEX SCHED block."""
        scan_info = self.vex["SCHED"][scan_nr]
        duration_str = scan_info["station"][2].split()[0]
        return int(duration_str)

    def start_time(self, scan_nr):
        """Extract the scan start time as an astropy Time object from the VEX SCHED block."""
        scan_info = self.vex["SCHED"][scan_nr]
        return parse_vex_time(scan_info["start"])

    def clock_offsets(self):
        """Extract per-station clock offsets [seconds].

        Returns:
            dict station -> offset_sec
        """
        offsets = {}
        for station in self.vex["STATION"].keys():
            station_offsets = self.clock_entry(station)[0]
            offsets[station] = station_offsets["offset_sec"]
        return offsets

    def clock_rates(self, scan_nr=None, at_time=None):
        """Extract per-station clock rates [sec/sec]."""
        rates = {}
        for station in self.vex["STATION"].keys():
            station_rates = self.clock_entry(station)[0]
            rates[station] = station_rates["rate_sec_per_sec"]
        return rates

    def center_frequencies(self):
        """Extract unique subband center frequencies in Hz from the VEX FREQ block.

        Adjusts each channel's edge frequency by half the bandwidth based on
        sideband (L=lower, U=upper) to compute the true center frequency.
        """
        all_chandefs = self.freq_def.getall('chan_def')

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

        return sorted(float(freq) * 1e6 for freq in center_frequencies)

    def channel_mapping(self):
        """Extract the VDIF thread-to-channel bit position mapping from the VEX THREADS block."""
        threads_block = self.vex['THREADS']
        thread_key = list(threads_block.keys())[0]
        thread_def = threads_block[thread_key]
        all_channels = thread_def.getall('channel')
        return [int(channel[2]) for channel in all_channels]

    def sample_rate(self):
        """Extract the sample rate from the VEX FREQ block.

        Returns:
            Sample rate in Hz (e.g. 32 Ms/sec -> 32000000.0).
        """
        sample_rate_str = self.freq_def['sample_rate']
        return float(sample_rate_str.split()[0]) * 1e6

    def subband_bandwidth(self):
        """Extract the subband bandwidth from the first channel definition in the VEX FREQ block.

        Returns:
            Subband bandwidth in Hz (e.g. 16 MHz -> 16000000.0).
        """
        all_chandefs = self.freq_def.getall('chan_def')
        bandwidth = float(all_chandefs[0][3].split()[0])
        return bandwidth * 1e6

    def stations(self):
        """Return the list of station names from the VEX STATION block."""
        return list(self.vex["STATION"].keys())

    def station_locations(self, station_names):
        """Extract geocentric EarthLocations for the given stations.

        Args:
            station_names: List of station identifiers (e.g. ['Ib', 'Ir']).

        Returns:
            Tuple of (site_names, locations) where site_names is a list of
            VEX SITE names and locations is a dict mapping station name to
            astropy EarthLocation.
        """
        site_names = []
        locations = {}
        for station in station_names:
            site = self.vex["STATION"][station]["SITE"]
            site_names.append(site)
            positions = self.vex["SITE"][site]["site_position"]
            coords = [float(pos.split()[0]) * un.m for pos in positions]
            locations[station] = ac.EarthLocation.from_geocentric(*coords)
        return site_names, locations

    def source_coords(self, scan_nr):
        """Extract the source sky coordinates for a given scan.

        Returns:
            astropy SkyCoord of the scan's target source.
        """
        scan_info = self.vex["SCHED"][scan_nr]
        source = self.vex["SOURCE"][scan_info["source"]]
        return ac.SkyCoord(
            ra=[source["ra"]],
            dec=[source["dec"]],
            frame="fk5",
            equinox="J2000",
        )
