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
from astropy.time import Time
from pycalc11 import Calc
from scipy.interpolate import CubicSpline


class CorrConfig:
    """
    Correlator configuration generator for ISBI-AARTFAAC correlator.

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

    def __extract_delays(self, ci: Calc, station_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract delay values from pycalc11 calculation results.

        Args:
            ci (pycalc11.Calc): Calc instance containing computed delays.
            station_list (List[str]): List of station names.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping station names to numpy arrays of delay values.
        """
        delays = {station: [] for station in station_list}

        for delay_set in ci.delay:
            for i, station in enumerate(station_list):
                delays[station].append(delay_set[0][i][0].value)

        return {station: np.array(vals) for station, vals in delays.items()}

    def __interpolate_delays(self, delays: Dict[str, np.ndarray], n_original: int, n_target: int) -> Dict[str, np.ndarray]:
        """
        Interpolate delays using cubic splines to match target sample count.

        Args:
            delays (Dict[str, np.ndarray]): Dictionary mapping station names to delay arrays.
            n_original (int): Number of original delay samples.
            n_target (int): Target number of delay samples after interpolation.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping station names to interpolated delay arrays.
        """
        x_original = np.linspace(0, 1, n_original)
        x_interp = np.linspace(0, 1, n_target)

        delays_interpolated = {}
        for station, delay_vals in delays.items():
            interp = CubicSpline(x_original, delay_vals)
            delays_interpolated[station] = interp(x_interp)

        return delays_interpolated

    def center_frequencies(self) -> np.ndarray:
        """
        Extract center frequencies for selected subbands from VEX file.

        Returns:
            numpy.ndarray: Array of center frequencies in MHz for selected subbands.
        """
        freq_key = next(iter(self.v["FREQ"].keys()))
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
            [all_center_freqs[sb - 1] for sb in self.subbands], dtype=np.float64
        )

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

        delays = self.__extract_delays(ci, station_list)

        duration_sec = int(duration_min * 60)
        n_samples = duration_sec // int(self.c["integration_time"]) + 1

        return self.__interpolate_delays(delays, len(ci.delay), n_samples)

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
        delays = [[-0.011373331712363283, -0.011373555850429876, -0.011373780011938036, -0.011374004196886207, -0.011374228405273554, -0.011374452637098536, -0.011374676892360313, -0.011374901171057335, -0.011375125473188667, -0.011375349798753048, -0.011375574147749269, -0.011375798520176134, -0.011376022916032465, -0.01137624733531697, -0.011376471778028736, -0.011376696244166191, -0.01137692073372851, -0.01137714524671414, -0.01137736978312225, -0.011377594342951287, -0.011377818926200229, -0.01137804353286807, -0.011378268162953254, -0.011378492816454935, -0.01137871749337157, -0.011378942193702321, -0.011379166917445727, -0.011379391664600585, -0.011379616435165978, -0.011379841229140363, -0.011380066046522872, -0.011380290887311977, -0.01138051575150664, -0.01138074063910586, -0.011380965550108089, -0.01138119048451247, -0.011381415442317475, -0.011381640423522231, -0.011381865428125214, -0.011382090456125477, -0.01138231550752174, -0.011382540582312813, -0.011382765680497488, -0.011382990802074583, -0.011383215947042884, -0.011383441115401207, -0.011383666307148255, -0.0113838915222831, -0.011384116760804194, -0.011384342022710691, -0.01138456730800104, -0.011384792616674407, -0.01138501794872922, -0.011385243304164554, -0.011385468682979127, -0.011385694085171734, -0.011385919510741182, -0.011386144959686267, -0.011386370432005712, -0.011386595927698575, -0.011386821446763295, -0.011387046989199049, -0.01138727255500426, -0.011387498144178098, -0.011387723756719008, -0.011387949392626046, -0.011388175051897938, -0.01138840073453347, -0.01138862644053145, -0.011388852169890672, -0.01138907792260994, -0.011389303698688062, -0.011389529498123732, -0.011389755320916036, -0.011389981167063388, -0.011390207036564967, -0.01139043292941921, -0.011390658845625182, -0.011390884785181592, -0.01139111074808724, -0.011391336734340929, -0.011391562743941446, -0.011391788776887594, -0.011392014833178173, -0.011392240912811892, -0.011392467015787812, -0.011392693142104383, -0.011392919291760747, -0.011393145464755366, -0.01139337166108737, -0.01139359788075521, -0.011393824123757945, -0.01139405039009429, -0.011394276679763038, -0.011394502992762987, -0.011394729329092941, -0.011394955688751682, -0.011395182071738006, -0.011395408478050638, -0.011395634907688627, -0.011395861360650414, -0.011396087836935158, -0.011396314336541289, -0.011396540859467888, -0.011396767405713633, -0.01139699397527734, -0.011397220568157793, -0.011397447184353798, -0.011397673823864142, -0.011397900486687621, -0.011398127172822955, -0.011398353882269181, -0.01139858061502475, -0.011398807371088813, -0.011399034150459808, -0.01139926095313688, -0.011399487779118473, -0.011399714628403642, -0.011399941500991108, -0.011400168396879642, -0.011400395316068053, -0.011400622258555032, -0.011400849224339562, -0.011401076213420428, -0.011401303225796163, -0.011401530261465837, -0.01140175732042813, -0.01140198440268193, -0.011402211508225771, -0.011402438637058617, -0.011402665789179445, -0.011402892964586683, -0.01140312016327949, -0.011403347385256288, -0.011403574630516234, -0.011403801899057757, -0.01140402919087992, -0.011404256505981425, -0.011404483844361062, -0.011404711206017621, -0.011404938590949907, -0.011405165999156688, -0.011405393430636773, -0.011405620885388853, -0.01140584836341198, -0.011406075864704608, -0.011406303389265869, -0.01140653093709421, -0.011406758508188668, -0.011406986102547954, -0.01140721372017086, -0.011407441361056174, -0.011407669025202688, -0.011407896712609182, -0.011408124423274454, -0.011408352157197219, -0.0114085799143765, -0.01140880769481075, -0.011409035498499106, -0.011409263325440008, -0.011409491175632596, -0.011409719049075303, -0.011409946945767193, -0.011410174865706951, -0.011410402808893369, -0.011410630775325236, -0.011410858765001341, -0.011411086777920387, -0.011411314814081419, -0.011411542873482875, -0.011411770956123909, -0.011411999062002923, -0.011412227191119083, -0.011412455343470811, -0.011412683519057169, -0.011412911717876836, -0.01141313993992862, -0.011413368185211313, -0.011413596453723588, -0.011413824745464506, -0.01141405306043259, -0.011414281398626623, -0.01141450976004566],
                  [-0.011374670439937515, -0.01137489449907904, -0.011375118581660425, -0.011375342687680132, -0.011375566817137315, -0.011375790970030442, -0.011376015146358666, -0.011376239346120445, -0.011376463569314842, -0.01137668781594058, -0.011376912085996477, -0.011377136379481328, -0.011377360696393942, -0.011377585036733045, -0.011377809400497716, -0.011378033787686382, -0.011378258198298222, -0.011378482632331691, -0.01137870708978594, -0.01137893157065943, -0.011379156074951135, -0.011379380602660057, -0.011379605153784625, -0.011379829728324005, -0.011380054326276656, -0.011380278947641733, -0.011380503592417776, -0.011380728260603587, -0.011380952952198249, -0.011381177667200214, -0.011381402405608618, -0.011381627167421934, -0.011381851952639136, -0.0113820767612592, -0.011382301593280587, -0.011382526448702449, -0.011382751327523248, -0.011382976229742126, -0.011383201155357538, -0.011383426104368554, -0.011383651076773894, -0.011383876072572352, -0.011384101091762742, -0.011384326134343873, -0.01138455120031453, -0.011384776289673524, -0.01138500140241957, -0.011385226538551738, -0.011385451698068474, -0.011385676880968938, -0.011385902087251577, -0.011386127316915554, -0.011386352569959306, -0.011386577846381904, -0.011386803146182064, -0.011387028469358584, -0.011387253815910268, -0.011387479185835924, -0.011387704579134265, -0.011387929995804347, -0.01138815543584462, -0.011388380899254246, -0.011388606386031666, -0.011388831896176033, -0.011389057429685804, -0.011389282986560039, -0.011389508566797456, -0.011389734170396843, -0.011389959797357008, -0.01139018544767675, -0.01139041112135487, -0.01139063681839017, -0.011390862538781356, -0.011391088282527505, -0.011391314049627049, -0.011391539840079139, -0.011391765653882233, -0.011391991491035388, -0.01139221735153732, -0.011392443235386823, -0.011392669142582707, -0.011392895073123754, -0.01139312102700877, -0.011393347004236548, -0.01139357300480581, -0.011393799028715613, -0.011394025075964402, -0.011394251146551328, -0.011394477240474842, -0.011394703357734083, -0.011394929498327495, -0.011395155662254151, -0.011395381849512758, -0.011395608060102105, -0.011395834294020997, -0.011396060551268242, -0.011396286831842605, -0.011396513135742912, -0.011396739462967858, -0.011396965813516504, -0.011397192187387302, -0.01139741858457939, -0.01139764500509122, -0.011397871448921855, -0.011398097916069995, -0.011398324406534434, -0.01139855092031397, -0.011398777457407402, -0.01139900401781352, -0.011399230601531127, -0.011399457208558936, -0.011399683838895986, -0.011399910492540724, -0.011400137169492312, -0.01140036386974918, -0.011400590593310483, -0.011400817340174651, -0.011401044110340746, -0.011401270903807494, -0.011401497720573662, -0.011401724560638055, -0.011401951423999374, -0.011402178310656597, -0.011402405220608511, -0.01140263215385365, -0.011402859110391081, -0.011403086090219484, -0.011403313093337761, -0.011403540119744421, -0.011403767169438455, -0.011403994242418818, -0.011404221338683948, -0.011404448458233012, -0.011404675601064422, -0.01140490276717734, -0.011405129956570198, -0.011405357169242049, -0.011405584405191606, -0.011405811664417656, -0.01140603894691899, -0.01140626625269441, -0.011406493581742692, -0.01140672093406263, -0.01140694830965293, -0.011407175708512645, -0.011407403130640219, -0.011407630576034791, -0.011407858044694809, -0.011408085536619308, -0.011408313051807001, -0.01140854059025668, -0.011408768151967133, -0.01140899573693715, -0.01140922334516552, -0.011409450976651034, -0.0114096786313924, -0.011409906309388656, -0.01141013401063825, -0.01141036173514032, -0.011410589482893304, -0.01141081725389634, -0.011411045048147873, -0.011411272865646955, -0.011411500706392278, -0.01141172857038263, -0.011411956457616805, -0.011412184368093593, -0.011412412301811687, -0.011412640258770148, -0.011412868238967403, -0.011413096242402608, -0.01141332426907417, -0.011413552318981257, -0.011413780392122281, -0.011414008488496311, -0.011414236608102037, -0.01141446475093825, -0.011414692917003741, -0.011414921106297193, -0.011415149318817671, -0.011415377554563691, -0.011415605813534041, -0.011415834095727776]]

        with open(config_path, "wb") as file:
            for i in range(len(delays)):
                file.write(struct.pack("i", len(delays[i])))
                file.write(struct.pack("d" * len(delays[i]), *delays[i]))

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
