import numpy as np
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


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


def shift_frame_selection(
        dataset_in,
        angles,
        reference_frame,
        shift_cutoff,
        recenter=False):

    shifts = []
    shifted_frames = np.zeros_like(dataset_in)

    for i in tqdm(range(dataset_in.shape[0])):
        tmp_shift = phase_cross_correlation(
            reference_frame,
            dataset_in[i, :, :],
            normalization=None,
            upsample_factor=10)[0]
        shifts.append(tmp_shift)

        shifted_frames[i, :, :] = shift(
            dataset_in[i, :, :],
            tmp_shift,
            order=5,
            mode="constant")

    # select based on cutoff
    distance_shift = np.sqrt(
        np.array(shifts)[:, 0] ** 2 +
        np.array(shifts)[:, 1] ** 2)

    keep_mask = distance_shift < shift_cutoff

    if recenter:
        return shifted_frames[keep_mask], angles[keep_mask]
    else:
        return dataset_in[keep_mask], angles[keep_mask]

