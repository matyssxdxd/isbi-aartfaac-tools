import argparse
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.process_data import process_data, read_visibility_file
from shutil import copyfile

class ISBIAARTFAACPlot:
    def __init__(self, filepaths, n_subbands, ctrl):
        self.filepaths = sorted(filepaths)
        self.n_subbands = n_subbands
        self.ctrl = ctrl
        self.raw_data = []
        self.read_raw_data()
        self.data = {
            'SCALAR': {
                'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
            },
            'VECTOR': {
                'BL0': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL0BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []},
                'BL1': {'RR': [], 'RL': [], 'LR': [], 'LL': []}
            },
        }
        self.scalar_average()
        self.vector_average()

    def read_raw_data(self, subband=None):
        if not subband:
            for file in self.filepaths:
                header, visibilities = read_visibility_file(file)
                self.raw_data.append(visibilities)
        else:
            file = self.filepaths[subband]
            header, visibilities = read_visibility_file(file)
            self.raw_data.append(visibilities)

    def scalar_average(self):
        for subband in range(self.n_subbands):
            raw_data = np.array(self.raw_data[subband], dtype=np.complex64)
            averaged_data = np.mean(raw_data, axis=0)
            averaged_data = np.swapaxes(averaged_data, 1, 2)
            for bidx, baseline in enumerate(['BL0', 'BL0BL1', 'BL1']):
                for pidx, pol in enumerate(['RR', 'RL', 'LR', 'LL']):
                    # Changed from extend to append to keep subbands separate
                    self.data['SCALAR'][baseline][pol].append(
                        averaged_data[bidx][pidx])

    def vector_average(self):
        for subband in range(self.n_subbands):
            raw_data = np.array(self.raw_data[subband], dtype=np.complex64)
            vec_averaged_data = np.sum(raw_data, axis=0)
            vec_averaged_data = np.swapaxes(vec_averaged_data, 1, 2)
            for bidx, baseline in enumerate(['BL0', 'BL0BL1', 'BL1']):
                for pidx, pol in enumerate(['RR', 'RL', 'LR', 'LL']):
                    # Changed from extend to append to keep subbands separate
                    self.data['VECTOR'][baseline][pol].append(
                        vec_averaged_data[bidx][pidx])

    def generate_plots(self, outdir, averaging='VECTOR'):
        """Generate individual PNG plots for each baseline, polarization, and subband"""
        plots = {}
        for baseline in ['BL0', 'BL0BL1', 'BL1']:
            plots[baseline] = {}
            for pol in ['RR', 'RL', 'LL', 'LR']:
                plots[baseline][pol] = {}
                
                # Retrieve list of data arrays (one per subband)
                data_list = self.data[averaging][baseline][pol]
                
                # Iterate over each subband's data
                for sb_idx, data_sb in enumerate(data_list):
                    # Only parallel hands
                    data = np.array(data_sb)
                    x_axis = np.arange(len(data))
                    
                    # Generate plot basename with subband suffix
                    basename = f"{baseline}_{pol}_sb{sb_idx}"
                    plots[baseline][pol][sb_idx] = {'basename': basename}

                    if baseline in ['BL0', 'BL1']:
                        # Auto-correlations
                        if pol in ['RR', 'LL']:
                            # Amplitude plot
                            fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                            ax.plot(x_axis, np.abs(data))
                            ax.set_xlabel('Channel', fontsize=8)
                            ax.set_ylabel('Amplitude', fontsize=8)
                            ax.set_title(f'{baseline} {pol} SB{sb_idx}', fontsize=9)
                            ax.grid(True, alpha=0.3)
                            ax.tick_params(labelsize=7)
                            plt.tight_layout()
                            plt.savefig(f'{outdir}/{basename}.png')
                            plt.close()

                            # Large version
                            fig, ax = plt.subplots(figsize=(10, 7))
                            ax.plot(x_axis, np.abs(data))
                            ax.set_xlabel('Channel')
                            ax.set_ylabel('Amplitude')
                            ax.set_title(f'{baseline} {pol} SB{sb_idx}')
                            ax.grid(True)
                            plt.tight_layout()
                            plt.savefig(f'{outdir}/{basename}_large.png')
                            plt.close()
                    else:
                        # Cross-correlations
                        # Amplitude plot (small)
                        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                        ax.plot(x_axis, np.abs(data))
                        ax.set_xlabel('Channel', fontsize=8)
                        ax.set_ylabel('Amplitude', fontsize=8)
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(labelsize=7)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_ampl.png')
                        plt.close()

                        # Amplitude plot (large)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        ax.plot(x_axis, np.abs(data))
                        ax.set_xlabel('Channel')
                        ax.set_ylabel('Amplitude')
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}')
                        ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_ampl_large.png')
                        plt.close()

                        # Phase plot (small)
                        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                        ax.plot(x_axis, np.angle(data, deg=True))
                        ax.set_xlabel('Channel', fontsize=8)
                        ax.set_ylabel('Phase (deg)', fontsize=8)
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(labelsize=7)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_ph.png')
                        plt.close()

                        # Phase plot (large)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        ax.plot(x_axis, np.angle(data, deg=True))
                        ax.set_xlabel('Channel')
                        ax.set_ylabel('Phase (deg)')
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}')
                        ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_ph_large.png')
                        plt.close()

                        # Lag spectrum (small)
                        lags = np.abs(np.fft.fftshift(np.fft.ifft(data)))
                        n = len(lags)
                        t = np.arange(-(n//2), n//2 + 1)
                        lag_offset = lags.argmax() - n//2
                        
                        plots[baseline][pol][sb_idx]['lag'] = lag_offset
                        plots[baseline][pol][sb_idx]['snr'] = lags.max() / np.median(lags)

                        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                        ax.plot(t, lags)
                        ax.set_xlabel('Lag', fontsize=8)
                        ax.set_ylabel('Amplitude', fontsize=8)
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(labelsize=7)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_lag.png')
                        plt.close()

                        # Lag spectrum (large)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        ax.plot(t, lags)
                        ax.set_xlabel('Lag')
                        ax.set_ylabel('Amplitude')
                        ax.set_title(f'{baseline} {pol} SB{sb_idx}')
                        ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(f'{outdir}/{basename}_lag_large.png')
                        plt.close()
        return plots

    def write_html(self, outdir, scan_name):
        """Generate HTML diagnostic page"""
        plots = self.generate_plots(outdir, averaging='VECTOR')

        htmlname = f"{outdir}/index.html"
        html = open(htmlname, 'w')

        exper_name = self.ctrl.get('exper_name', 'Unknown')

        # HTML HEAD
        html.write("<html><head>\n"
            + f"  <title>ISBI Correlator output - {exper_name} </title>\n"
            + "  <style> BODY,TH,TD{font-size: 10pt } \n"
            + "    .popup_img a { position:relative; }\n"
            + "    .popup_img a img { position:absolute; display:none; top:20; height:200; z-index:99;}\n"
            + "    .popup_img a:hover img { display:block; }\n"
            + "</style>\n"
            + "</head> <body>\n")

        # Print preamble
        html.write(f"Scan name = {scan_name}<br>\n")
        html.write(f"Number of subbands = {self.n_subbands}<br><br>\n")

        # Loop over each subband to create a separate table
        for sb_idx in range(self.n_subbands):
            html.write(f"<h3>Subband {sb_idx}</h3>\n")
            html.write("<div class='popup_img'>\n")
            html.write("<table border=1 bgcolor='#dddddd' cellspacing=0>\n")

            # Header row
            html.write("<tr>\n" 
                + f"  <th rowspan=2> {exper_name} </th>\n"
                + "  <th colspan=2>Auto correlations</th>\n"
                + "  <th colspan=1>Cross correlations (SNR, lag offset)</th>\n"
                + "</tr>\n")

            # Second row
            html.write("<tr>\n")
            html.write("<th>BL0</th><th>BL1</th><th>BL0-BL1</th>\n")
            html.write("</tr>\n")

            # Data rows - one per polarization
            for pol in ['RR', 'RL', 'LL', 'LR']:
                html.write("<tr>\n")
                html.write(f"<th>{pol}</th>\n")

                if pol in ['RR', 'LL']:
                    # Auto BL0
                    basename = plots['BL0'][pol][sb_idx]['basename']
                    html.write("  <td>" 
                        + f"<a href='{basename}_large.png'>Ib"
                        + f"<img src='{basename}.png' />"
                        + "</a></td>\n")

                    # Auto BL1
                    basename = plots['BL1'][pol][sb_idx]['basename']
                    html.write("  <td>" 
                        + f"<a href='{basename}_large.png'>Ir"
                        + f"<img src='{basename}.png' />"
                        + "</a></td>\n")
                else:
                    html.write("  <td>Cross-hands</td>")
                    html.write("  <td>Cross-hands</td>")

                # Cross BL0BL1
                try:
                    basename = plots['BL0BL1'][pol][sb_idx]['basename']
                    snr = plots['BL0BL1'][pol][sb_idx]['snr']
                    offset = plots['BL0BL1'][pol][sb_idx]['lag']

                    # Color code by SNR
                    if snr >= 6:
                        green = min(int(100 + 155 * (snr-6) / 3.), 255)
                        colour = f"#00{green:02x}00"
                    else:
                        red = min(int(100 + 155 * (6-snr) / 3.), 255)
                        colour = f"#{red:02x}0000"

                    html.write(f"  <td bgcolor='{colour}'>")
                    html.write(f"<a href='{basename}_lag_large.png'>"
                        + f"<img src='{basename}_lag.png' />"
                        + f"{snr:.1f}</a>")
                    html.write(f"<a href='{basename}_ampl_large.png' >"
                        + f"<img src='{basename}_ampl.png' />"
                        + "A</a>")
                    html.write(f"<a href='{basename}_ph_large.png' >"
                        + f"<img src='{basename}_ph.png' />"
                        + f"P</a><br><font size=-2>offset: {offset}</font>")
                    html.write("</td>\n")
                except KeyError:
                    html.write("  <td> <br> </td>\n")

                html.write("</tr>\n")

            html.write("</table>\n")
            html.write("</div>\n<br>\n")

        html.write("</body></html>\n")
        html.close()

        print(f"HTML diagnostic page written to: {htmlname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ctrl", help="Control file", required=True)
    parser.add_argument("-s", "--scan", help="Scan name", required=True)
    args = parser.parse_args()

    with open(args.ctrl, 'r') as ctrl:
        ctrl_f = json.load(ctrl)

    # Get output files for this scan
    # out_files = glob.glob(f'{ctrl_f["output-path"]}{ctrl_f["exper_name"]}/{args.scan}/*.out')
    out_files = glob.glob(f'{ctrl_f["data-path"][7:]}/*.out')
    n_subbands = len(ctrl_f['subbands'])

    print(f"Processing {n_subbands} subbands for scan {args.scan}")

    # Create output directory for HTML
    htmldir = f'{ctrl_f["output-path"]}{ctrl_f["exper_name"]}/{args.scan}/html'
    os.makedirs(htmldir, exist_ok=True)

    # Generate plots and HTML
    plot = ISBIAARTFAACPlot(out_files, n_subbands, ctrl_f)
    plot.write_html(htmldir, args.scan)

