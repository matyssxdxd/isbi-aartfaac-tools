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

import matplotlib.pyplot as plt

import sys

class CorrConfig:
    """
    Correlator configuration generator for ISBI-AARTFAAC correlator.
#
    Attributes:
        v (vex.Vex): Parsed VEX file as vex class.
        c (Dict[str, Any]): Parsed JSON configuration file.
        subbands (List[int]): Selected subbands to process.
        n_subbands (int): Number of subbands to process.
        observation_name (str): Name of the experiment.
        output_path (str): Output directory path for generated configuration files.
    """

    v: vex.Vex
    c: Dict[str, Any]
    subbands: List[int]
    n_subbands: int
    observation_name: str
    output_path: str

    def __init__(self, vex_path: str, ctrl_path: str) -> None:
        """
        Initialize CorrConfig with VEX and control file paths.

        Args:
            vex_path (str): Path to the VEX observation file.
            ctrl_path (str): Path to the control JSON configuration file.
        """
        self.v = self.__load_vex(vex_path)
        self.c = self.__load_ctrl(ctrl_path)

        self.subbands = self.c["subbands"]
        self.n_subbands = len(self.subbands)
        self.observation_name = self.c["exper_name"]
        self.output_path = f"{self.c['output-path']}{self.observation_name}"

        os.makedirs(self.output_path, exist_ok=True)

    def __load_vex(self, path: str) -> vex.Vex:
        """
        Load and parse VEX observation file.

        Args:
            path (str): Path to the VEX file.

        Returns:
            vex.Vex: Parsed VEX file as vex class.
        """
        with open(path) as f:
            return vex.parse(f.read())

    def __load_ctrl(self, path: str) -> Dict[str, Any]:
        """
        Load control configuration from JSON file.

        Args:
            path (str): Path to the control JSON file.

        Returns:
            Dict[str, Any]: Control configuration dictionary.
        """
        with open(path) as f:
            return json.load(f)

    def __parse_vex_time(self, vex_time: str) -> Time:
        """
        Parse VEX time format to astropy Time object.

        Args:
            vex_time (str): Time string in VEX format (e.g., '2024y289d13h43m47s').

        Returns:
            astropy.time.Time: Parsed time in UTC scale.
        """
        year = int(vex_time[:4])
        day_of_year = int(vex_time[5:8])
        hour = int(vex_time[9:11])
        minute = int(vex_time[12:14])
        second = int(vex_time[15:17])

        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        formatted_date = f"{date.strftime('%Y-%m-%d')}T{hour:02}:{minute:02}:{second:02}.000"

        return Time(formatted_date, format="isot", scale="utc")

    def __station_locations(self) -> Tuple[Dict[str, ac.EarthLocation], List[str]]:
        """
        Get ordered station locations with reference station first.

        Returns:
            Tuple[Dict[str, ac.EarthLocation], List[str]]: A tuple containing:
                - Dictionary mapping station names to EarthLocation objects.
                - List of station site names.
        """
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

    def center_frequencies(self) -> np.ndarray:
        """
        Extract sky center frequencies for selected subbands.
        One frequency per subband (not per channel).

        Returns:
            numpy.ndarray: Array of sky center frequencies in MHz, one per subband.
        """
        freq_key = next(iter(self.v["FREQ"].keys()))
        chan_def = self.v["FREQ"][freq_key].getall("chan_def")
        bandwidth_mhz = float(chan_def[0][3].split()[0])  # 16.00 MHz

        all_subband_freqs = []

        for i in range(0, len(chan_def), 2):  # Step by 2 (RCP + LCP = 1 subband)
            bbc_freq_mhz = float(chan_def[i][1].split()[0])
            sideband = chan_def[i][2]  # 'L' or 'U'

            if sideband == 'L':
                sky_freq = bbc_freq_mhz - bandwidth_mhz / 2
            elif sideband == 'U':
                sky_freq = bbc_freq_mhz + bandwidth_mhz / 2
            else:
                raise ValueError(f"Unknown sideband: {sideband}")

            all_subband_freqs.append(sky_freq)

        selected_freqs = [all_subband_freqs[sb - 1] for sb in self.subbands]

        return np.array(selected_freqs, dtype=np.float64)

    def channel_mapping(self) -> np.ndarray:
        """
        Generate channel mapping for selected subbands.

        Returns:
            numpy.ndarray: Array of channel mapping indices (uint32).
        """
        threads_key = next(iter(self.v["THREADS"].keys()))
        threads = self.v["THREADS"][threads_key].getall("channel")

        all_chan_mapping = [int(thread[-1]) for thread in threads]

        channel_mapping = []
        for subband in self.subbands:
            start_idx = (subband - 1) * 2
            channel_mapping.extend(all_chan_mapping[start_idx : start_idx + 2])

        return np.array(channel_mapping, dtype=np.uint32)

    def scan_start_time(self, scan: str) -> Time:
        """
        Get the start time for a specific scan.

        Args:
            scan (str): Scan identifier from VEX SCHED section.

        Returns:
            astropy.time.Time: Scan start time in UTC.
        """
        date_str = self.v["SCHED"][scan]["start"]
        return self.__parse_vex_time(date_str)

    def scan_delays(self, scan: str) -> Dict[str, np.ndarray]:
        """
        Calculate geometric delays for all stations during a scan.

        Args:
            scan (str): Scan identifier from VEX SCHED section.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping station names to interpolated delay arrays.
        """
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

        n_samples = 181
        duration_sec = duration_min * 60

        time_offsets = np.linspace(0, duration_sec, n_samples)
        fine_time_grid = start_time + TimeDelta(time_offsets, format='sec')
        high_res_delays = ci.interpolate_delays(fine_time_grid)

        delays = {station: [] for station in station_list}
        g_delays = {station: [] for station in station_list}

        # clock_offset = {'Ir': 8.233828125e-6, 'Ib': 8.114e-6}
        # clock_rate = {'Ir': 2.02e-07, 'Ib': 1.99e-07}
        # clock_epoch = self.__parse_vex_time("2024y289d14h37m08s")
        
        # clock_offset = {'Ir': 5.2140625e-6, 'Ib': 5.019e-6}
        # clock_rate = {'Ir': -1.79e-07, 'Ib': 1.61e-07}
        # clock_epoch = self.__parse_vex_time("2024y103d05h01m32")
        
        clock_offset = {'Ir':1.3999999999999993e-05, 'Ib':0}
        clock_rate = {'Ir': -2.0249999999999998e-05, 'Ib': 0}
        clock_epoch = start_time
        
        for t_idx, current_time in enumerate(fine_time_grid):
            dt_seconds = (current_time - start_time).sec
            geometric_delay = high_res_delays[t_idx]

            for i, station in enumerate(station_list):
                drift = clock_offset[station] + dt_seconds * clock_rate[station]
                geo_delay = geometric_delay[0][i][0].value

                total_delay = geo_delay + drift

                g_delays[station].append(geo_delay)
                delays[station].append(total_delay)
                
        for i in range(180):
            print(f"{i}, total_delay_ir={delays['Ir'][i]}, total_delay_ib={delays['Ib'][i]}")
            print(f"{i}, geometric_delay_ir={g_delays['Ir'][i]}, geometric_delay_ib={g_delays['Ib'][i]}")
            g_diff = g_delays["Ir"][i] - g_delays["Ib"][i]
            print(f"{i}, geometretic ir - ib in samples={g_diff*32e6}")
            t_diff = delays["Ir"][i] - delays["Ib"][i]
            print(f"{i}, total ir - ib in samples={t_diff*32e6}")
        
        return delays

    def write_scan_config(self, scan: str) -> None:
        """
        Write binary configuration file for a scan.

        Args:
            scan (str): Scan identifier from VEX SCHED section.
        """
        config_path = f"{self.output_path}/{scan}/{scan}.conf"

        delays = self.scan_delays(scan)
        center_frequencies = self.center_frequencies()
        channel_mapping = self.channel_mapping()
        print(center_frequencies)
        print(channel_mapping)
        with open(config_path, "wb") as file:
            for delay in delays:
                print(delay)
                file.write(struct.pack("i", len(delays[delay])))
                file.write(struct.pack("d" * len(delays[delay]), *delays[delay]))

            file.write(struct.pack("i", len(center_frequencies)))
            file.write(struct.pack("d" * len(center_frequencies), *center_frequencies))

            file.write(struct.pack("i", len(channel_mapping)))
            file.write(struct.pack("i" * len(channel_mapping), *channel_mapping))

    def scan_run_cmd(self, scan: str) -> str:
        """
        Generate correlator run command for a scan.

        Args:
            scan (str): Scan identifier from VEX SCHED section.

        Returns:
            str: Complete command line string for running the correlator.
        """
        config_path = f"{self.output_path}/{scan}/{scan}.conf"
        channels = int(self.c["number_of_channels"])
        start_time = self.__parse_vex_time(self.v["SCHED"][scan]["start"])
        runtime = int(self.v["SCHED"][scan]["station"][2].split()[0])

        freq_key = next(iter(self.v["FREQ"].keys()))
        sample_rate = (
            float(self.v["FREQ"][freq_key].get("sample_rate").split()[0]) * un.MHz
        )
        bandwidth = (
            float(self.v["FREQ"][freq_key].get("chan_def")[3].split()[0]) * un.MHz
        )

        samples_per_second = sample_rate.to(un.Hz).value
        samples_per_subband = (
            int(self.c["integration_time"]) // 2 * int(samples_per_second / channels)
        )

        # Build input paths
        ref_station = self.c["reference-station"]
        input_paths = [self.c["data"][ref_station][scan]]
        input_paths.extend(
            self.c["data"][station][scan]
            for station in self.c["data"]
            if station != ref_station
        )
        input_path = ",".join(input_paths)

        # Build output paths
        base_out = f"{self.output_path}/{scan}/"
        output_paths = [
            f"{base_out}subband_{sb}.out" for sb in self.subbands
        ]
        output_path = ",".join(output_paths)

        cmd = (
            f"TZ=UTC ISBI/ISBI --configFile {config_path} -n2 "
            f"-t{samples_per_subband} -c{channels} -C{channels-1} -b16 -s{self.n_subbands} "
            f'-m15 -D "{str(start_time).replace("T", " ")[:-4]}" -r{runtime} '
            f"-g0 -q1 -R0 -B0 -f{samples_per_second} "
            f"--subbandBandwidth {bandwidth.to(un.Hz).value} -i {input_path} -o {output_path}"
        )

        return cmd

    def get_scan_config(self, scan: str) -> None:
        """
        Generate and display scan configuration.

        Args:
            scan (str): Scan identifier from VEX SCHED section.
        """
        scan_path = f"{self.output_path}/{scan}"
        os.makedirs(scan_path, exist_ok=True)

        self.write_scan_config(scan)
        print(self.scan_run_cmd(scan))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate correlator configuration for ISBI-AARTFAAC"
    )
    parser.add_argument("vex", help="Path to VEX observation file")
    parser.add_argument("control", help="Path to control JSON configuration file")
    parser.add_argument("scan", help="Scan identifier from VEX SCHED section")

    args = parser.parse_args()

    conf = CorrConfig(args.vex, args.control)
    conf.get_scan_config(args.scan)
