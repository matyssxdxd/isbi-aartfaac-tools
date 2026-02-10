"""VEXtractor: extract observation parameters from a parsed VEX file.

Provides methods to read station, frequency, clock, and scheduling
information from a VEX observation file.
"""

import vex

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
        """Extract clock early offsets in seconds for each station from the VEX CLOCK block."""
        clock_block = self.vex['CLOCK']
        clock_offsets = {}
        for station in clock_block:
            clock_early = clock_block[station].get('clock_early')
            offset_usec = float(clock_early[1].split()[0])
            clock_offsets[station] = offset_usec * 1e-6
        return clock_offsets

    def clock_rates(self):
        """Extract clock drift rates for each station from the VEX CLOCK block."""
        clock_block = self.vex['CLOCK']
        clock_rates = {}
        for station in clock_block:
            clock_early = clock_block[station].get('clock_early')
            clock_rates[station] = float(clock_early[3])
        return clock_rates

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
