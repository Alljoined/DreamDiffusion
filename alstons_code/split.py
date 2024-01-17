import numpy as np

def split_npy(input_file):
    data = np.load(input_file)
    x_dim = data.shape[0]

    for i in range(x_dim):
        split = data[i, :, :]
        output_file = f'split_{i + 1}.npy'
        np.save(output_file, split)

# Example usage:
input_file = 'zhou2016.npy'
split_npy(input_file)
