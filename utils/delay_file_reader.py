import struct
import numpy as np

class DelayFileReader:
    def __init__(self, filename):
        self.filename = filename
        self.header = {}
        self.scans = []

    def read_file(self):
        with open(self.filename, 'rb') as f:
            self.read_header(f)
            self.read_scans(f)
        return self.header, self.scans

    def read_header(self, file):
        # Header consists of header_size (int32), version (int32), n_padding_seconds (int32), and station name (3 chars)
        self.header["header_size"] = struct.unpack('i', file.read(4))[0]
        self.header["version"] = struct.unpack('i', file.read(4))[0]
        self.header["n_padding_seconds"] = struct.unpack('i', file.read(4))[0]
        self.header["station_name"] = file.read(3).decode('utf-8').strip('\x00')

    def read_scans(self, file_handle):
        current_scan = None

        while True:
            try:
                # Try to read a scan name (81 chars)
                scan_name_bytes = file_handle.read(81)
                if not scan_name_bytes or len(scan_name_bytes) < 81:
                    break

                scan_name = scan_name_bytes.decode('utf-8').strip('\x00')

                # Read source name (81 chars)
                source_name = file_handle.read(81).decode('utf-8').strip('\x00')

                # Read MJD (int32)
                scan_mjd = struct.unpack('i', file_handle.read(4))[0]

                current_scan = {
                    'scan_name': scan_name,
                    'source_name': source_name,
                    'mjd': scan_mjd,
                    'points': []
                }
                self.scans.append(current_scan)

                # Read points until the marker for the end of the scan
                while True:
                    # Read a data point
                    data = file_handle.read(7 * 8)  # 7 doubles (8 bytes each)
                    if len(data) < 7 * 8:
                        return

                    values = struct.unpack('ddddddd', data)
                    sec_of_day, u, v, w, delay, phase, amplitude = values

                    # Check if this is an end marker (all zeros)
                    if all(v == 0 for v in values):
                        break

                    # Add the point to the current scan
                    current_scan['points'].append({
                        'sec_of_day': sec_of_day,
                        'uvw': (u, v, w),
                        'delay': delay,
                        'phase': phase,
                        'amplitude': amplitude
                    })

            except Exception as e:
                print(f"Error reading scan: {e}")
                break