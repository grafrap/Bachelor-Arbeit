import h5py

def parse_mps_data(file_path):
    mps_data = []
    current_mps = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('['):
                if current_mps:
                    mps_data.append(current_mps)
                current_mps = {'id': line, 'data': []}
            elif line and current_mps is not None:
                current_mps['data'].append(line)
        if current_mps:
            mps_data.append(current_mps)
    return mps_data

def write_mps_to_hdf5(mps_data, hdf5_file_path):
    with h5py.File(hdf5_file_path, 'w') as h5file:
        for i, mps in enumerate(mps_data):
            group = h5file.create_group(f"state_{i+1}")
            group.attrs['id'] = mps['id']
            for j, data_line in enumerate(mps['data']):
                dataset_name = f"data_{j+1}"
                group.create_dataset(dataset_name, data=data_line)

def main():
    input_file_path = 'output.txt'  # Path to the file containing MPS data
    hdf5_file_path = 'mps_data.h5'  # Path to the HDF5 file to be created

    # Parse the MPS data from the text file
    mps_data = parse_mps_data(input_file_path)

    # Write the parsed MPS data to the HDF5 file
    write_mps_to_hdf5(mps_data, hdf5_file_path)

if __name__ == "__main__":
    main()