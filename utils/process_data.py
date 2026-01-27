import numpy as np
import struct
import sys

class Header:
    def __init__(self):
        self.magic = None
        self.nr_receivers = None
        self.nr_polarizations = None
        self.correlation_mode = None
        self.start_time = None
        self.end_time = None
        self.weights = None
        self.nr_samples_per_integration = None
        self.nr_channels = None
        self.pad0 = None
        self.first_channel_frequency = None
        self.channel_bandwidth = None
        self.pad1 = None

def read_visibility_file(visibility_path):
    headers = []
    visibilities = []
    
    with open(visibility_path, 'rb') as file:
        while True:
            try:
                # Read header
                header = Header()
                header.magic = struct.unpack('I', file.read(4))[0]
                header.nr_receivers = struct.unpack('H', file.read(2))[0]
                header.nr_polarizations = struct.unpack('B', file.read(1))[0]
                header.correlation_mode = struct.unpack('B', file.read(1))[0]
                header.start_time = struct.unpack('d', file.read(8))[0]
                header.end_time = struct.unpack('d', file.read(8))[0]
                header.weights = struct.unpack('I' * 300, file.read(4 * 300))
                header.nr_samples_per_integration = struct.unpack('I', file.read(4))[0]
                header.nr_channels = struct.unpack('H', file.read(2))[0]
                header.pad0 = file.read(2)
                header.first_channel_frequency = struct.unpack('d', file.read(8))[0]
                header.channel_bandwidth = struct.unpack('d', file.read(8))[0]
                header.pad1 = file.read(288)
                
                vis_dtype = np.complex64
                nr_baselines = header.nr_receivers + \
                    int(header.nr_receivers * (header.nr_receivers - 1) / 2)
                vis_shape = (nr_baselines, header.nr_channels, 2, 2)
                vis_zeros = np.zeros(vis_shape, vis_dtype)
                
                # Read visibilities
                vis = file.read(vis_zeros.size * vis_zeros.itemsize)
                if len(vis) < vis_zeros.size * vis_zeros.itemsize:
                    break
                    
                vis = np.frombuffer(vis, dtype=vis_dtype).reshape(vis_shape)
                
                headers.append(header)
                visibilities.append(vis)
                
            except struct.error:
                break
    
    return headers, visibilities

def average_visibilities(visibilities):
    vis_array = np.array(visibilities, dtype=np.complex64)
    averaged_visibilities = np.mean(vis_array, axis=0)
    averaged_visibilities = np.swapaxes(averaged_visibilities, 1, 2)

    # median_amp = np.median(np.abs(averaged_visibilities))
    # if median_amp > 0:  # Avoid division by zero
    #     averaged_visibilities = averaged_visibilities / median_amp
    return averaged_visibilities

def vector_sum_visibilities(visibilities):
    vis_array = np.array(visibilities, dtype=np.complex64)
    summed_visibilities = np.sum(vis_array, axis=0)
    vector_average = summed_visibilities
    vector_average = np.swapaxes(vector_average, 1, 2)
    return vector_average


def average_integrations(visibilities):
    result = []
    for vis in visibilities:
        avg = np.mean(vis, axis=1)
        result.append(avg)
    return np.array(result, dtype=np.complex64)

def process_data(corr_files, integration_average=False, vector_average=False):
    processed_visibilities = []

    for file in corr_files:
        headers, visibilities = read_visibility_file(file)
        if integration_average:
            processed_visibilities.append(average_integrations(visibilities))
        elif vector_average:
            processed_visibilities.append(vector_sum_visibilities(visibilities))
        else:
            processed_visibilities.append(average_visibilities(visibilities))

    return processed_visibilities
