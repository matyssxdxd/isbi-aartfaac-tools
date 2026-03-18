#!/usr/bin/env python3
"""Generate ISBI-AARTFAAC correlator run command and delay configuration file.

Parses a control JSON file and VEX observation file to extract correlator
parameters, compute geometric delays, and produce the shell command to run
the ISBI-AARTFAAC correlator.
"""

import numpy as np
import argparse
import json
import os

from utils.vextractor import VEXtractor
from utils.pycalc11_delays import pycalc11_delays
from utils.helpers import parse_arguments, save_config
from utils.sfxc_delays import sfxc_delays

DEBUG = True

def generate_run_cmd(config_path, nr_samples_per_channel, nr_channels, subbands, start_time,
                     runtime, sample_rate, subband_bandwidth, input_path, output_path):
    """Generate the shell command to run the ISBI-AARTFAAC correlator.

    Args:
        config_path: Path to the delay/frequency configuration file (.conf).
        nr_samples_per_channel: Number of time samples per frequency channel. Must be a multiple of 16.
        nr_channels: Number of FFT channels per subband.
        subbands: List of 1-indexed subband numbers to process.
        start_time: Observation start time in 'YYYY-MM-DD H:M:S' format.
        runtime: Scan duration in seconds.
        sample_rate: Sampling rate in Hz.
        subband_bandwidth: Subband bandwidth in Hz.
        input_path: Comma-separated input file paths, reference station first.
        output_path: Comma-separated output file paths, one per subband.

    Returns:
        The correlator run command as a string.
    """
    run_cmd = (f'TZ=UTC ISBI/ISBI --configFile {config_path} -p1 -n2 '
               f'-t{nr_samples_per_channel} -c{nr_channels} -C{nr_channels - 1} '
               f'-b16 -s{len(subbands)} -m15 '
               f'-D "{start_time}" -r{runtime} '
               f'-g0 -q1 -R0 -B0 '
               f'-f{sample_rate} --subbandBandwidth {subband_bandwidth} '
               f'-i {input_path} -o {output_path}')

    return run_cmd


if __name__ == "__main__":
    description = 'Generate ISBI-AARTFAAC run cmd and configuration file'
    arguments = {
        'control': 'Path to control JSON file'
    }

    args = parse_arguments(description, arguments)

    with open(args.control) as f:
        ctrl_file = json.load(f)

    vex = VEXtractor(ctrl_file['vex-path'])

    scan_nr = ctrl_file['scan-number']
    subbands = ctrl_file['subbands']
    integration_time = ctrl_file['integration-time']
    nr_channels = ctrl_file['nr-channels']
    reference_station = ctrl_file['reference-station']

    duration = vex.duration(scan_nr)
    start_time = vex.start_time(scan_nr)
    sample_rate = int(vex.sample_rate())
    subband_bandwidth = vex.subband_bandwidth()
    center_frequencies = vex.center_frequencies()
    channel_mapping = vex.channel_mapping()

    nr_samples_per_channel = (int(sample_rate * integration_time) // (nr_channels * 2)) 
    nr_samples_per_channel -= nr_samples_per_channel % 8
    n_integrations = np.ceil(duration / ((nr_samples_per_channel * nr_channels * 2) / sample_rate)) + 1

    delay_type = ctrl_file['delay-type']

    if delay_type == 'sfxc':
        delay_paths = {station: delay_path for station, delay_path in ctrl_file['delay-paths'].items()}
        delays = sfxc_delays(vex, delay_paths, scan_nr, integration_time, n_integrations, reference_station)
    else:
        delays = pycalc11_delays(vex, scan_nr, reference_station=reference_station)

    selected_indices = []
    for subband in subbands:
        selected_indices.extend([2 * (subband - 1), 2 * (subband - 1) + 1])
    center_frequencies = [center_frequencies[i - 1] for i in subbands]
    channel_mapping = [channel_mapping[i] for i in selected_indices]

    if not DEBUG:
        full_output_path = f'{ctrl_file["output-path"]}/{ctrl_file["experiment"]}/{scan_nr}'
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)

        config_path = f'{full_output_path}/{scan_nr}.conf'
    else:
        full_output_path = './'
        config_path = f'{full_output_path}/{scan_nr}.conf'

    save_config(config_path, delays, center_frequencies, channel_mapping)

    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    data_paths = ctrl_file['data-paths']
    other_stations = [s for s in data_paths if s != reference_station]
    input_path = ','.join([data_paths[reference_station]] + [data_paths[s] for s in other_stations])

    if not DEBUG:
        output_path = ','.join(f'{full_output_path}/subband_{s}.out' for s in subbands)
    else:
        full_output_path = f'{ctrl_file["output-path"]}/{ctrl_file["experiment"]}/{scan_nr}'
        config_path = f'{full_output_path}/{scan_nr}.conf'
        output_path = ','.join(f'{full_output_path}/subband_{s}.out' for s in subbands)

    run_cmd = generate_run_cmd(
        config_path=config_path,
        nr_samples_per_channel=nr_samples_per_channel,
        nr_channels=nr_channels,
        subbands=subbands,
        start_time=start_time_str,
        runtime=duration,
        sample_rate=sample_rate,
        subband_bandwidth=subband_bandwidth,
        input_path=input_path,
        output_path=output_path,
    )

    print(run_cmd)
