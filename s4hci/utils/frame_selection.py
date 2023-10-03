import numpy as np


def mse_frame_selection(dataset_in,
                        angles,
                        percent_cutoff):

    if percent_cutoff == 0:
        return dataset_in, angles

    mean_frame = np.mean(dataset_in, axis=0)
    residual_sequence = (dataset_in - mean_frame) ** 2
    mse_sequence = np.mean(residual_sequence, axis=(1, 2))

    cutoff = int(len(mse_sequence) / 100 * percent_cutoff)
    remove_index = np.argsort(mse_sequence)[-cutoff:]

    new_frames = np.delete(dataset_in, remove_index, axis=0)
    new_angles = np.delete(angles, remove_index, axis=0)

    return new_frames, new_angles
