#!/usr/bin/env python3
"""Generate ISBI-AARTFAAC correlator run command and delay configuration file.

Parses a control JSON file and VEX observation file to extract correlator
parameters, compute geometric delays, and produce the shell command to run
the ISBI-AARTFAAC correlator.

Control file keys (required):
    experiment, vex-path, scan-number, output-path, subbands,
    reference-station, integration-time, nr-channels, data-paths,
    delay-type ("sfxc" or "pycalc11"), delay-paths (sfxc only)

Optional keys:
    stations            Ordered station list. Default: reference station
                        first, then the rest in data-paths order. This ONE list
                        orders both -i and the delay tables, which must agree.
    channel-mapping     {station: "4,6,0,2,5,7,1,3"} -- VDIF channel per
                        (subband, pol). Overrides the mapping derived from the
                        vex $THREADS. Required for any station whose vex has
                        no $THREADS (n26l2), or whose recorder does not write
                        channels in the order its vex claims.
    clock-offsets-usec  {station: usec} added to the vex clock. For a vex with
                        no $CLOCK (n26l2) this is the whole clock, from a
                        fringe search.
    reference-only      true -> --referenceStation 0: only baselines with the
                        reference station (plus all autocorrelations).
    start-time          "YYYY-MM-DD HH:MM:SS" to correlate part of a scan.
    runtime             Seconds, with start-time.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from utils.vextractor import VEXtractor
from utils.pycalc11_delays import pycalc11_delays
from utils.helpers import save_config
from utils.sfxc_delays import sfxc_delays

# ISBI's tensor-core correlator compiles with NR_TIMES_PER_BLOCK = 128 / bits,
# and -t must be a multiple of it or the NVRTC compile fails. -b16 -> 8.
NR_BITS = 16
NR_TIMES_PER_BLOCK = 128 // NR_BITS


def generate_run_cmd(config_path, center_frequencies, nr_samples_per_channel, nr_channels, subbands, start_time,
                     runtime, sample_rate, subband_bandwidth, input_path, output_path, nr_stations,
                     channel_mapping, max_delay, station_sample_rates=None, reference_only=False):
    """Generate the shell command to run the ISBI-AARTFAAC correlator.

    Args:
        config_path: Path to the delay file (--delayFile).
        center_frequencies: Subband sky centre frequencies in Hz (-F).
        nr_samples_per_channel: Time samples per channel per block (-t). Must be
            a multiple of NR_TIMES_PER_BLOCK (8 for -b16).
        nr_channels: Number of FFT channels per subband.
        subbands: List of 1-indexed subband numbers to process.
        start_time: Observation start time in 'YYYY-MM-DD H:M:S' format.
        runtime: Duration in seconds.
        sample_rate: The correlator's -f in Hz: the LOWEST station rate.
        subband_bandwidth: Subband bandwidth in Hz.
        input_path: Comma-separated input file paths, in station order.
        output_path: Comma-separated output file paths, one per subband.
        nr_stations: -n; must equal the number of inputs.
        channel_mapping: Per-station lists of mapping entries, station order.
        max_delay: --maxDelay in samples: the largest |differential delay|.
        station_sample_rates: Per-station rates in Hz, only when they differ;
            faster stations are decimated by ISBI to sample_rate.
        reference_only: Pass --referenceStation 0.

    Returns:
        The correlator run command as a string.
    """
    mapping = ";".join(",".join(entries) for entries in channel_mapping)
    run_cmd = (f'TZ=UTC ISBI/ISBI --delayFile {config_path} -F {",".join(str(x) for x in center_frequencies)} '
               f'-p1 -n{nr_stations} '
               f'-t{nr_samples_per_channel} -c{nr_channels} -C{nr_channels - 1} '
               f'-b{NR_BITS} -s{len(subbands)} -m15 '
               f'-D "{start_time}" -r{runtime} '
               f'-g0 -q1 -R0 -B0 -d1 '
               f'-f{sample_rate} --subbandBandwidth {subband_bandwidth} '
               f'--channelMapping "{mapping}" --maxDelay {max_delay} ')
    if station_sample_rates is not None:
        run_cmd += f'--stationSampleRate "{";".join(str(r) for r in station_sample_rates)}" '
    if reference_only:
        run_cmd += '--referenceStation 0 '
    run_cmd += f'-i {input_path} -o {output_path}'

    return run_cmd


def ordered_stations(ctrl):
    """The one station order used for -i, the delay tables and the mapping."""
    reference = ctrl["reference-station"]
    stations = ctrl.get("stations") or (
        [reference] + [s for s in ctrl["data-paths"] if s != reference])
    if stations[0] != reference:
        raise SystemExit(f"'stations' must start with the reference station '{reference}' -- "
                         "ISBI subtracts the FIRST delay table from all the others")
    missing = [s for s in stations if s not in ctrl["data-paths"]]
    if missing:
        raise SystemExit(f"No data-paths entry for {missing}")
    return stations


def max_delay_samples(delays, stations, sample_rate):
    """Largest |delay(station) - delay(reference)| in samples, with margin,
    never below ISBI's own default of 1000.

    ISBI starts reading this far before the first block and seeks by it, so it
    must cover the largest differential delay in the table. The default of 1000
    suits E011's co-located Ib/Ir; n26l2's Ef-Hh reaches ~97k samples.

    The floor matters for reproducibility, not correctness: the header weight
    counts valid samples over a window that reaches maxDelay before each block,
    so a different maxDelay changes the weight (125003 vs 125004 for E011)
    while the visibilities stay bit-identical.
    """
    reference = delays[stations[0]]["delay"]
    largest = 0.0
    for station in stations[1:]:
        d = delays[station]["delay"]
        n = min(len(d), len(reference))
        largest = max(largest, float(np.max(np.abs(d[:n] - reference[:n]))))
    return max(1000, int(math.ceil(largest * sample_rate * 1.1)) + 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ISBI-AARTFAAC run cmd and delay file")
    parser.add_argument("control", help="Path to control JSON file")
    parser.add_argument("--conf-dir", default=None,
                        help="Directory for the .conf delay file (default: the run's output directory)")
    args = parser.parse_args()

    with open(args.control) as f:
        ctrl_file = json.load(f)

    vex = VEXtractor(ctrl_file["vex-path"])

    scan_nr = ctrl_file["scan-number"]
    subbands = ctrl_file["subbands"]
    integration_time = ctrl_file["integration-time"]
    nr_channels = ctrl_file["nr-channels"]
    stations = ordered_stations(ctrl_file)
    reference_station = stations[0]

    # --- sample rates: -f is the slowest station; faster ones are decimated ---
    rates = [int(vex.sample_rate(s)) for s in stations]
    sample_rate = min(rates)
    for station, rate in zip(stations, rates):
        if rate % sample_rate != 0:
            raise SystemExit(f"{station} records at {rate} Hz, not an integer multiple of {sample_rate} Hz; "
                             "ISBI can only decimate by an integer factor")
    if rates[0] != sample_rate:
        raise SystemExit(f"The reference station {reference_station} records at {rates[0]} Hz, but the "
                         f"correlator runs at the slowest rate {sample_rate} Hz. Use a station at that rate "
                         "as reference: its subbands define the correlator's frequency grid.")
    station_sample_rates = rates if len(set(rates)) > 1 else None

    # --- frequency grid: the reference station's subbands ---------------------
    subband_bandwidth = vex.subband_bandwidth(reference_station)
    all_centers = vex.center_frequencies(reference_station)
    center_frequencies = [all_centers[i - 1] for i in subbands]

    # --- channel mapping, per station, per (subband, pol) ---------------------
    overrides = ctrl_file.get("channel-mapping", {})
    channel_mapping = []
    for station in stations:
        if station in overrides:
            entries = overrides[station]
            entries = [e.strip() for e in entries.split(",")] if isinstance(entries, str) else [str(e) for e in entries]
        else:
            entries = vex.channel_mapping(station, center_frequencies)
            if entries is None:
                raise SystemExit(f"The vex gives no $THREADS for {station}, so its VDIF channel order is unknown. "
                                 f"Give it in the control file: \"channel-mapping\": {{\"{station}\": \"...\"}}, "
                                 f"{2 * len(subbands)} entries, one per (subband, pol).")
        if len(entries) != 2 * len(subbands):
            raise SystemExit(f"channel mapping for {station} has {len(entries)} entries; "
                             f"{len(subbands)} subbands x 2 pols needs {2 * len(subbands)}")
        channel_mapping.append(entries)

    # --- time window ------------------------------------------------------------
    if "start-time" in ctrl_file:
        start_time_str = ctrl_file["start-time"]
        runtime = int(ctrl_file["runtime"])
    else:
        start_time_str = vex.start_time(scan_nr).strftime("%Y-%m-%d %H:%M:%S")
        runtime = vex.duration(scan_nr)

    nr_samples_per_channel = int(sample_rate * integration_time) // (nr_channels * 2)
    rounded = nr_samples_per_channel - nr_samples_per_channel % NR_TIMES_PER_BLOCK
    if rounded != nr_samples_per_channel:
        print(f"note: -t {nr_samples_per_channel} rounded down to {rounded}, a multiple of "
              f"{NR_TIMES_PER_BLOCK}; integration is {rounded * nr_channels * 2 / sample_rate:.6f} s",
              file=sys.stderr)
        nr_samples_per_channel = rounded

    # --- delays -----------------------------------------------------------------
    clock_offsets_usec = ctrl_file.get("clock-offsets-usec", {})
    for station in stations:
        if not vex.has_clock(station) and station not in clock_offsets_usec and station != reference_station:
            print(f"warning: no vex clock for {station} and no clock-offsets-usec entry; using 0",
                  file=sys.stderr)

    if ctrl_file["delay-type"] == "sfxc":
        delays = sfxc_delays(vex, ctrl_file["delay-paths"], scan_nr, stations, sample_rate,
                             clock_offsets_usec)
    else:
        delays = pycalc11_delays(vex, scan_nr, stations, sample_rate, clock_offsets_usec)

    full_output_path = os.path.abspath(f'{ctrl_file["output-path"]}/{ctrl_file["experiment"]}/{scan_nr}')
    conf_dir = os.path.abspath(args.conf_dir) if args.conf_dir else full_output_path
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(full_output_path, exist_ok=True)
    config_path = f"{conf_dir}/{scan_nr}.conf"

    save_config(config_path, delays, stations)

    data_paths = ctrl_file["data-paths"]
    input_path = ",".join(data_paths[s] for s in stations)
    output_path = ",".join(f"{full_output_path}/subband_{s}.out" for s in subbands)

    run_cmd = generate_run_cmd(
        config_path=config_path,
        center_frequencies=center_frequencies,
        nr_samples_per_channel=nr_samples_per_channel,
        nr_channels=nr_channels,
        subbands=subbands,
        start_time=start_time_str,
        runtime=runtime,
        sample_rate=sample_rate,
        subband_bandwidth=subband_bandwidth,
        input_path=input_path,
        output_path=output_path,
        nr_stations=len(stations),
        channel_mapping=channel_mapping,
        max_delay=max_delay_samples(delays, stations, sample_rate),
        station_sample_rates=station_sample_rates,
        reference_only=bool(ctrl_file.get("reference-only", False)),
    )

    print(run_cmd)
