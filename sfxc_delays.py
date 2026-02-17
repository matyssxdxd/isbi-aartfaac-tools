import numpy as np
import os
from vextractor import VEXtractor, parse_vex_time
from generate_delays import geometric_delays, save_config
from utils.delay_file_reader import DelayFileReader
from scipy.interpolate import Akima1DInterpolator

def read_delays(file, scan_name):
    reader = DelayFileReader(file)
    reader.read_file()

    matching_scans = [scan for scan in reader.scans if scan['scan_name'] == scan_name]
    if not matching_scans:
        raise ValueError(f"Scan '{scan_name}' not found in {file}")

    scan = matching_scans[0]
    sec_of_day = []
    delays = []
    for point in scan['points']:
        sec_of_day.append(point['sec_of_day'])
        delays.append(point['delay'])

    return np.array(sec_of_day), np.array(delays)

if __name__ == "__main__":
    EXPER = 'E011'
    SCAN = 'No0002'
    DELAY_FILE_IR = './E011/E011_Ir.del'
    DELAY_FILE_IB = './E011/E011_Ib.del'
    VEX_PATH = '/home/matyss/Work/RADIOBLOCKS/isbi-aartfaac-tools/E011/E011.vix'
    OUTPUT_PATH = f'/home/matyss/Work/RADIOBLOCKS/{EXPER}/'
    SUBBANDS = [1, 2, 3, 4, 5, 6, 7, 8]

    vex = VEXtractor(VEX_PATH)

    duration = vex.duration(SCAN)
    n_integrations = 91
    g_delays, g_time_offsets = geometric_delays(vex, SCAN, n_integrations=n_integrations)

    clock_offsets = vex.clock_offsets()   # {'IR': 1.85e-07, 'IB': 0.0}
    clock_rates = vex.clock_rates()       # {'IR': -1.66e-07, 'IB': 2.37e-13}

    scan_start = vex.start_time(SCAN)
    clock_epoch = parse_vex_time('2024y121d03h51m15s')
    epoch_offset = (clock_epoch - scan_start).sec  # seconds from scan start to clock epoch

    # Read raw geometric delays from SFXC delay tables
    ib_sod, ib_del = read_delays(DELAY_FILE_IB, SCAN)
    ir_sod, ir_del = read_delays(DELAY_FILE_IR, SCAN)

    # Interpolate to your target times using Akima splines
    scan_start_sod = (scan_start.mjd % 1) * 86400.0
    target_sod = scan_start_sod + g_time_offsets

    ib_interp = Akima1DInterpolator(ib_sod, ib_del)(target_sod)
    ir_interp = Akima1DInterpolator(ir_sod, ir_del)(target_sod)

    # Add clock model: offset + rate * (time - clock_epoch)
    for i, t_offset in enumerate(g_time_offsets):
        dt = t_offset - epoch_offset  # seconds from clock_epoch
        ib_interp[i] += clock_offsets['Ib'] + dt * clock_rates['Ib']
        ir_interp[i] += clock_offsets['Ir'] + dt * clock_rates['Ir']

    sfxc_delays = {
        'Ib': ib_interp,
        'Ir': ir_interp
    }

    center_frequencies = vex.center_frequencies()
    channel_mapping = vex.channel_mapping()

    selected_indices = []
    for subband in SUBBANDS:
        selected_indices.extend([2 * (subband - 1), 2 * (subband - 1) + 1])

    center_frequencies = [center_frequencies[i - 1] for i in SUBBANDS]
    channel_mapping = [channel_mapping[i] for i in selected_indices]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)


    save_config(
        f"{OUTPUT_PATH}{SCAN}.conf",
        sfxc_delays,
        center_frequencies,
        channel_mapping
    )

    print(sfxc_delays)
    print(center_frequencies)
    print(channel_mapping)