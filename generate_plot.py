import argparse
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils.process_data import process_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ctrl", help="Control file")

    args = parser.parse_args()

    with open(args.ctrl, 'r') as ctrl:
        ctrl_f = json.load(ctrl)

    out_files = glob.glob(f'{ctrl_f["data-path"][7:]}/*.out')
    n_subbands = ctrl_f['subbands']
    data = {
        'AUTO': {
            'BL0': {
                'RR': [],
                'LL': []
            },
            'BL1': {
                'RR': [],
                'LL': []
            }
        },
        'CROSS': {
            'BL0BL1': {
                'RR': [],
                'RL': [],
                'LR': [],
                'LL': []
            }
        }
    }

    baselines = ['BL0', 'BL0BL1', 'BL1']
    polarizations = ['RR', 'RL', 'LR', 'LL']

    vis = process_data(out_files)
    for subband in range(n_subbands):
        # Auto-correlations
        for baseline in [0, 2]:
            curr_bl = baselines[baseline]
            for pol in ['RR', 'LL']:
                pol_index = polarizations.index(pol)
                data['AUTO'][curr_bl][pol].extend(vis[subband][baseline][pol_index][:])

        # Cross-correlation
        baseline = 1
        curr_bl = baselines[baseline]
        for pol in polarizations:
            pol_index = polarizations.index(pol)
            data['CROSS'][curr_bl][pol].extend(vis[subband][baseline][pol_index][:])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
    fig.suptitle(ctrl_f['plot-name'])

    x_axis = np.arange(len(data['AUTO']['BL0']['RR']))

    # AUTO correlations
    ax1.plot(x_axis, np.abs(data['AUTO']['BL0']['RR']), label='RR')
    ax1.plot(x_axis, np.abs(data['AUTO']['BL0']['LL']), label='LL')
    ax1.set_title('AUTO BL0')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_axis, np.abs(data['AUTO']['BL1']['RR']), label='RR')
    ax2.plot(x_axis, np.abs(data['AUTO']['BL1']['LL']), label='LL')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('AUTO BL1')
    ax2.legend()
    ax2.grid(True)

    # CROSS correlation phase (degrees)
    ax3.scatter(x_axis, np.angle(data['CROSS']['BL0BL1']['RR'], deg=True), label='RR', s=10)
    ax3.scatter(x_axis, np.angle(data['CROSS']['BL0BL1']['RL'], deg=True), label='RL', s=10)
    ax3.scatter(x_axis, np.angle(data['CROSS']['BL0BL1']['LR'], deg=True), label='LR', s=10)
    ax3.scatter(x_axis, np.angle(data['CROSS']['BL0BL1']['LL'], deg=True), label='LL', s=10)
    ax3.set_title('CROSS BL0BL1')
    ax3.set_ylabel('Phase (deg)')
    ax3.set_xlabel('Sample index')
    ax3.legend()
    ax3.grid(True)

    # CROSS correlation
    ax4.plot(x_axis, np.abs(data['CROSS']['BL0BL1']['RR']), label='RR')
    ax4.plot(x_axis, np.abs(data['CROSS']['BL0BL1']['RL']), label='RL')
    ax4.plot(x_axis, np.abs(data['CROSS']['BL0BL1']['LR']), label='LR')
    ax4.plot(x_axis, np.abs(data['CROSS']['BL0BL1']['LL']), label='LL')
    ax4.set_title('CROSS BL0BL1')
    ax4.set_ylabel('Amplitude')
    ax4.set_xlabel('Sample index')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    data = {
        'CROSS': {
            'BL0BL1': {
                'RR': [],
                'RL': [],
                'LR': [],
                'LL': []
            }
        }
    }

    baseline = 1
    curr_bl = baselines[baseline]

    vis = process_data(out_files, True) 
    for subband in range(len(vis)):
        for integ in range(vis[subband].shape[0]):
            data['CROSS'][curr_bl]['RR'].append(vis[subband][integ][baseline][0])
            data['CROSS'][curr_bl]['RL'].append(vis[subband][integ][baseline][1])
            data['CROSS'][curr_bl]['LR'].append(vis[subband][integ][baseline][2])
            data['CROSS'][curr_bl]['LL'].append(vis[subband][integ][baseline][3])

    rr_arr = np.array(data['CROSS'][curr_bl]['RR'])
    rl_arr = np.array(data['CROSS'][curr_bl]['RL'])
    lr_arr = np.array(data['CROSS'][curr_bl]['LR'])
    ll_arr = np.array(data['CROSS'][curr_bl]['LL'])

    integration_time = 2  # seconds
    num_integrations = len(rr_arr) // 8   # = 712 // 8 = 89

    time = np.arange(num_integrations) * integration_time
    time_full = np.tile(time, 8)
    x = np.arange(len(rr_arr)) % num_integrations * integration_time

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    fig.suptitle(ctrl_f['plot-name'])
    ax1.scatter(x, np.angle(rr_arr, deg=True), label='RR', s=8)
    # ax1.scatter(x, np.angle(rl_arr, deg=True), label='RL', s=8)
    # ax1.scatter(x, np.angle(lr_arr, deg=True), label='LR', s=8)
    ax1.scatter(x, np.angle(ll_arr, deg=True), label='LL', s=8)
    ax1.set_ylim([-200, 200])
    ax1.set_ylabel('Phase (deg)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, np.abs(rr_arr), label='RR')
    # ax2.plot(x, np.abs(rl_arr), label='RL')
    # ax2.plot(x, np.abs(lr_arr), label='LR')
    ax2.plot(x, np.abs(ll_arr), label='LL')
    ax2.set_ylim([-0.1 * 1e8, 2 * 1e8])
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
