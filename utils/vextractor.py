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

    date = Time(f"{year}-01-01T00:00:00.000", format="isot", scale="utc") + TimeDelta(
        day_of_year - 1, format="jd"
    )
    formatted_date = (
        f"{date.strftime('%Y-%m-%d')}T{hour:02}:{minute:02}:{second:02}.000"
    )
    return Time(formatted_date, format="isot", scale="utc")


class VEXtractor:
    """Extract observation parameters from a parsed VEX file.

    Args:
        vex_path: Path to the VEX file on disk.
    """

    def __init__(self, vex_path):
        with open(vex_path) as f:
            self.vex = vex.parse(f.read())

        # The first $FREQ def, for callers that do not name a station. Only
        # correct when every station shares one mode -- n26l2 has four FREQ
        # defs (8x32 MHz, 2x64 MHz, ...), so prefer the per-station methods.
        freq_block = self.vex["FREQ"]
        freq_key = list(freq_block.keys())[0]
        self.freq_def = freq_block[freq_key]

    def _mode_ref(self, block, station):
        """Name of the `block` def ($FREQ, $THREADS, $BBC, $IF, ...) that $MODE
        assigns to `station`, or None if no mode assigns one."""
        for mode in self.vex["MODE"]:
            try:
                refs = self.vex["MODE"][mode].getall(block)
            except KeyError:
                continue
            for ref in refs:
                if station in list(ref)[1:]:
                    return ref[0]
        return None

    def _freq_def_for(self, station=None):
        if station is None:
            return self.freq_def
        ref = self._mode_ref("FREQ", station)
        if ref is None:
            raise KeyError(f"No $FREQ def assigned to station '{station}' in $MODE")
        return self.vex["FREQ"][ref]

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
        """Return parsed clock_early entries for a station, oldest first.

        Each entry is a dict with keys: start, offset_sec, rate_sec_per_sec, epoch.
        Returns an empty list when the vex has no $CLOCK section or the station
        has no CLOCK reference -- n26l2's vex has neither, and its clocks have to
        come from a fringe search instead (control file "clock-offsets-usec").
        """
        if station not in self.vex["STATION"]:
            raise KeyError(f"Unknown station '{station}'")

        if "CLOCK" not in self.vex:
            return []
        try:
            clock_ref = self.vex["STATION"][station]["CLOCK"]
        except KeyError:
            return []
        clock_block = self.vex["CLOCK"][clock_ref]

        entries = clock_block.getall("clock_early")

        if not entries:
            raise ValueError(
                f"No clock_early entries for station '{station}' (CLOCK='{clock_ref}')"
            )

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

    def clock_entry(self, station, at_time=None):
        """Return the clock_early entry in force for a station at `at_time`.

        A vex may hold several clock_early entries per station, each valid from
        its own start time onwards (a new maser offset after a reset, or a
        per-scan fringe-fit result). The one in force at a scan is the LATEST
        entry starting at or before it. With no `at_time` the earliest entry is
        returned, which is what this used to do unconditionally.

        Returns None when the station has no clock_early at all.
        """
        entries = self._clock_entries_for_station(station)
        if not entries:
            return None
        if at_time is None:
            return entries[0]

        at_time = self._as_time(at_time)
        valid = [e for e in entries if e["start"] <= at_time]
        # Before the first validity start: use the first entry rather than none.
        return valid[-1] if valid else entries[0]

    def duration(self, scan_nr):
        """Extract the scan duration in seconds from the VEX SCHED block."""
        scan_info = self.vex["SCHED"][scan_nr]
        duration_str = scan_info["station"][2].split()[0]
        return int(duration_str)

    def start_time(self, scan_nr):
        """Extract the scan start time as an astropy Time object from the VEX SCHED block."""
        scan_info = self.vex["SCHED"][scan_nr]
        return parse_vex_time(scan_info["start"])

    def clock_offsets(self, at_time=None):
        """Per-station clock offsets [seconds] in force at `at_time`.

        Stations with no clock_early get 0, so a vex without $CLOCK still works;
        their real offsets must then be supplied separately.
        """
        offsets = {}
        for station in self.vex["STATION"].keys():
            entry = self.clock_entry(station, at_time)
            offsets[station] = entry["offset_sec"] if entry else 0.0
        return offsets

    def clock_rates(self, at_time=None):
        """Per-station clock rates [sec/sec] in force at `at_time` (0 if none)."""
        rates = {}
        for station in self.vex["STATION"].keys():
            entry = self.clock_entry(station, at_time)
            rates[station] = entry["rate_sec_per_sec"] if entry else 0.0
        return rates

    def clock_epoch(self, at_time=None):
        """Per-station clock-rate reference epochs in force at `at_time`.

        A station with no clock_early gets `at_time` (or None), which makes its
        rate term vanish since its rate is 0 as well.
        """
        epochs = {}
        for station in self.vex["STATION"].keys():
            entry = self.clock_entry(station, at_time)
            epochs[station] = entry["epoch"] if entry else (self._as_time(at_time) if at_time is not None else None)
        return epochs

    def has_clock(self, station):
        """True if the vex gives this station any clock_early entry."""
        return bool(self._clock_entries_for_station(station))

    @staticmethod
    def _chan_center_hz(chan_def):
        """Sky centre of one chan_def: its edge frequency moved half a bandwidth
        down for a lower sideband, up for an upper one."""
        freq = float(chan_def[1].split()[0])
        bandwidth = float(chan_def[3].split()[0])
        if chan_def[2] == "L":
            freq -= bandwidth / 2
        elif chan_def[2] == "U":
            freq += bandwidth / 2
        return freq * 1e6

    def center_frequencies(self, station=None):
        """Extract unique subband center frequencies in Hz from the VEX FREQ block.

        Adjusts each channel's edge frequency by half the bandwidth based on
        sideband (L=lower, U=upper) to compute the true center frequency.
        With `station`, uses that station's own $FREQ def.
        """
        all_chandefs = self._freq_def_for(station).getall("chan_def")

        center_frequencies = set()
        for chan_def in all_chandefs:
            freq = float(chan_def[1].split()[0])
            bound = chan_def[2]
            bandwidth = float(chan_def[3].split()[0])

            if bound == "L":
                freq -= bandwidth / 2
            elif bound == "U":
                freq += bandwidth / 2

            center_frequencies.add(freq)

        return sorted(float(freq) * 1e6 for freq in center_frequencies)

    def freqs(self):
        all_chandefs = self.freq_def.getall("chan_def")
        freqs = []
        for chan_def in all_chandefs:
            freq = float(chan_def[1].split()[0])
            freqs.append(freq)

        return freqs

    def _chan_polarizations(self, station):
        """chan link -> polarisation letter ('R', 'L', 'X', 'Y') for a station,
        resolved through chan_def -> $BBC BBC_assign -> $IF if_def. Returns {}
        if the vex does not give enough to resolve it."""
        try:
            bbc_ref = self._mode_ref("BBC", station)
            if_ref = self._mode_ref("IF", station)
            if bbc_ref is None or if_ref is None:
                return {}
            bbc_to_if = {row[0]: row[2] for row in self.vex["BBC"][bbc_ref].getall("BBC_assign")}
            if_to_pol = {row[0]: row[2] for row in self.vex["IF"][if_ref].getall("if_def")}
            return {cd[4]: if_to_pol.get(bbc_to_if.get(cd[5]))
                    for cd in self._freq_def_for(station).getall("chan_def")}
        except KeyError:
            return {}

    def channel_mapping(self, station, subband_centers_hz=None):
        """ISBI --channelMapping entries for one station, per (subband, pol).

        For each subband (sorted sky centre, or the given `subband_centers_hz`)
        the station's two chan_defs at that centre are found, ordered R before L
        (X before Y) by their actual polarisation, and each is looked up in the
        station's $THREADS def for its VDIF (thread, channel). An entry is the
        channel number, or "thread.channel" when the thread is not 0.

        Pairing by polarisation rather than by chan_def position matters: if a
        station listed L before R, adjacency would correlate R against L.

        Returns None when the station has no $THREADS def. n26l2's vex has only
        $TRACKS, and its fanout claims CH01..CH08 are in order -- which
        measurement showed is wrong for that recorder -- so such stations must
        be given their mapping explicitly ("channel-mapping" in the control file).
        """
        threads_ref = self._mode_ref("THREADS", station)
        if threads_ref is None:
            return None
        vdif = {row[0]: (int(row[1]), int(row[2]))
                for row in self.vex["THREADS"][threads_ref].getall("channel")}

        pols = self._chan_polarizations(station)
        order = {"R": 0, "X": 0, "L": 1, "Y": 1}

        by_center = {}
        for position, cd in enumerate(self._freq_def_for(station).getall("chan_def")):
            center = round(self._chan_center_hz(cd))
            by_center.setdefault(center, []).append(
                (order.get(pols.get(cd[4]), position), position, cd[4]))

        centers = (sorted(by_center) if subband_centers_hz is None
                   else [round(c) for c in subband_centers_hz])

        entries = []
        for center in centers:
            if center not in by_center:
                raise ValueError(f"Station '{station}' has no channel centred at {center / 1e6:.4f} MHz")
            chans = sorted(by_center[center])
            if len(chans) != 2:
                raise ValueError(f"Station '{station}' has {len(chans)} channels at "
                                 f"{center / 1e6:.4f} MHz; ISBI needs exactly 2 polarisations")
            for _, _, link in chans:
                if link not in vdif:
                    raise ValueError(f"Channel {link} of station '{station}' is not in $THREADS '{threads_ref}'")
                thread, channel = vdif[link]
                entries.append(str(channel) if thread == 0 else f"{thread}.{channel}")
        return entries

    def sample_rate(self, station=None):
        """Extract the sample rate from the VEX FREQ block.

        With `station`, uses that station's own $FREQ def; stations in one
        experiment can record at different rates (n26l2: 64 vs 128 Ms/s).

        Returns:
            Sample rate in Hz (e.g. 32 Ms/sec -> 32000000.0).
        """
        sample_rate_str = self._freq_def_for(station)["sample_rate"]
        return float(sample_rate_str.split()[0]) * 1e6

    def subband_bandwidth(self, station=None):
        """Extract the subband bandwidth from the first channel definition in the VEX FREQ block.

        Returns:
            Subband bandwidth in Hz (e.g. 16 MHz -> 16000000.0).
        """
        all_chandefs = self._freq_def_for(station).getall("chan_def")
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
