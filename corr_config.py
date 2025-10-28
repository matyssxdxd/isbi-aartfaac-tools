import argparse
import json
import os
import struct
from datetime import datetime, timedelta

import numpy as np
from astropy import coordinates as ac
from astropy import units as un
from astropy.time import Time
from pycalc11 import Calc
from scipy.interpolate import CubicSpline

import vex


class CorrConfig:
    """
    Yeah
    """

    def __init__(self, vex_path, ctrl_path):
        self.v = self.__load_vex(vex_path)
        self.c = self.__load_ctrl(ctrl_path)

        self.subbands = self.c["subbands"]
        self.n_subbands = len(self.subbands)
        self.observation_name = self.c["exper_name"]
        self.output_path = f"{self.c['output-path']}{self.observation_name}"

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)

    def __load_vex(self, path):
        """Loads observations VEX file."""
        with open(path) as f:
            return vex.parse(f.read())

    def __load_ctrl(self, path):
        """Loads control JSON file."""
        with open(path) as f:
            return json.load(f)

    def __parse_vex_time(self, vex_time):
        """Parse VEX time format (e.g., '2024y289d13h43m47s') to astropy Time."""
        year = int(vex_time[:4])
        day_of_year = int(vex_time[5:8])
        hour = int(vex_time[9:11])
        minute = int(vex_time[12:14])
        second = int(vex_time[15:17])

        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        formatted_date = date.strftime(
            f"%Y-%m-%dT{hour:02}:{minute:02}:{second:02}.000"
        )

        return Time(formatted_date, format="isot", scale="utc")

    def __station_locations(self):
        """Get ordered station locations with reference station first."""
        reference_station = self.c["reference-station"]
        other_stations = [s for s in self.c["stations"] if s != reference_station]
        station_names = [reference_station] + other_stations

        station_sites = [
            self.v["STATION"][station]["SITE"] for station in station_names
        ]

        station_locations = {}
        for station in station_names:
            site = self.v["STATION"][station]["SITE"]
            positions = self.v["SITE"][site]["site_position"]
            coords = [float(pos.split()[0]) * un.m for pos in positions]
            station_locations[station] = ac.EarthLocation.from_geocentric(*coords)

        return station_locations, station_sites

    def __extract_delays(self, ci, station_list):
        """Extract delay values from pycalc11 results."""
        delays = {station: [] for station in station_list}

        for delay_set in ci.delay:
            for i, station in enumerate(station_list):
                delays[station].append(delay_set[0][i][0].value)

        return {station: np.array(vals) for station, vals in delays.items()}

    def __interpolate_delays(self, delays, n_original, n_target):
        """Interpolate delays to get the exact number of delays required."""
        x_original = np.linspace(0, 1, n_original)
        x_interp = np.linspace(0, 1, n_target)

        delays_interpolated = {}
        for station, delay_vals in delays.items():
            interp = CubicSpline(x_original, delay_vals)
            delays_interpolated[station] = interp(x_interp)

        return delays_interpolated

    def center_frequencies(self):
        freq_key = list(self.v["FREQ"].keys())[0]
        chan_def = self.v["FREQ"][freq_key].getall("chan_def")

        all_center_freqs = []
        for i in range(0, len(chan_def), 2):
            frequency = float(chan_def[i][1].split()[0])
            bandwidth = float(chan_def[i][3].split()[0])
            bound = chan_def[i][2]

            offset = bandwidth / 2
            center_freq = frequency + offset if bound == "U" else frequency - offset
            all_center_freqs.append(center_freq)

        return np.array(
            [all_center_freqs[sb - 1] for sb in self.subbands], dtype=np.double
        )

    def channel_mapping(self):
        threads_key = list(self.v["THREADS"].keys())[0]
        threads = self.v["THREADS"][threads_key].getall("channel")

        all_chan_mapping = [int(thread[-1]) for thread in threads]

        channel_mapping = []
        for subband in self.subbands:
            start_idx = (subband - 1) * 2
            channel_mapping.extend(all_chan_mapping[start_idx : start_idx + 2])

        return np.array(channel_mapping, dtype=np.uint32)

    def scan_start_time(self, scan):
        date_str = self.v["SCHED"][scan]["start"]
        return self.__parse_vex_time(date_str)

    def scan_delays(self, scan):
        station_locations, station_sites = self.__station_locations()
        station_list = list(station_locations.keys())

        scan_info = self.v["SCHED"][scan]
        source = self.v["SOURCE"][scan_info["source"]]
        source_coords = ac.SkyCoord(
            [source["ra"]], [source["dec"]], frame="fk5", equinox="J2000.0"
        )

        start_time = self.__parse_vex_time(scan_info["start"])
        duration_min = int(scan_info["station"][2].split()[0]) / 60

        ci = Calc(
            station_names=station_sites,
            station_coords=list(station_locations.values()),
            source_coords=source_coords,
            start_time=start_time,
            duration_min=duration_min,
        )
        ci.run_driver()

        delays = self.__extract_delays(ci, station_list)

        duration_sec = int(duration_min * 60)
        n_samples = duration_sec // int(self.c["integration_time"]) + 1

        return self.__interpolate_delays(delays, len(ci.delay), n_samples)

    def write_scan_config(self, scan):
        config_path = f"{self.output_path}/{scan}/{scan}.conf"

        delays = self.scan_delays(scan)
        center_frequencies = self.center_frequencies()
        channel_mapping = self.channel_mapping()

        with open(config_path, "wb") as file:
            for delay in delays:
                file.write(struct.pack("i", len(delays[delay])))
                file.write(struct.pack("d" * len(delays[delay]), *delays[delay]))

            file.write(struct.pack("i", len(center_frequencies)))
            file.write(struct.pack("d" * len(center_frequencies), *center_frequencies))

            file.write(struct.pack("i", len(channel_mapping)))
            file.write(struct.pack("i" * len(channel_mapping), *channel_mapping))

    def scan_run_cmd(self, scan):
        config_path = f"{self.output_path}/{scan}/{scan}.conf"
        channels = int(self.c["number_of_channels"])
        start_time = self.__parse_vex_time(self.v["SCHED"][scan]["start"])
        runtime = int(self.v["SCHED"][scan]["station"][2].split()[0])
        freq_key = list(self.v["FREQ"].keys())[0]
        sample_rate = (
            float(self.v["FREQ"][freq_key].get("sample_rate").split()[0]) * un.MHz
        )
        bandwidth = (
            float(self.v["FREQ"][freq_key].get("chan_def")[3].split()[0]) * un.MHz
        )
        sample_time = float(1.0 / sample_rate.to(un.Hz).value)
        samples_per_second = 1.0 / sample_time
        samples_per_subband = (
            int(self.c["integration_time"]) // 2 * int(samples_per_second / channels)
        )

        input_path = self.c["data"][self.c["reference-station"]][scan]
        for station in self.c["data"]:
            if station != self.c["reference-station"]:
                input_path += f",{self.c['data'][station][scan]}"

        base_out = f"{self.output_path}/{scan}/"
        output_path = ""
        for i in range(1, self.n_subbands + 1):
            output_path += f"{base_out}subband_{self.subbands[i-1]}.out"
            if i != self.n_subbands:
                output_path += ","

        cmd = (
            f"TZ=UTC ISBI/ISBI --configFile {config_path} -n2 "
            f"-t{samples_per_subband} -c{channels} -C{channels-1} -b16 -s{self.n_subbands} "
            f'-m15 -D "{str(start_time).replace("T", " ")[:-4]}" -r{runtime} '
            f"-g0 -q1 -R0 -B0 -f{sample_rate.to(un.Hz).value} "
            f"--subbandBandwidth {bandwidth.to(un.Hz).value} -i {input_path} -o {output_path}"
        )

        return cmd

    def get_scan_config(self, scan):
        scan_path = f"{self.output_path}/{scan}"

        if not os.path.isdir(scan_path):
            os.mkdir(scan_path)

        self.write_scan_config(scan)
        print(self.scan_run_cmd(scan))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vex")
    parser.add_argument("control")

    args = parser.parse_args()

    conf = CorrConfig(args.vex, args.control)
    conf.get_scan_config("No0002")
