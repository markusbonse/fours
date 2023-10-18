import numpy as np
import multiprocessing
from scipy.ndimage import rotate
from tqdm import tqdm


def cadi_psf_subtraction(
        images: np.ndarray,
        angles: np.ndarray):

    median_frame = np.median(images, axis=0)
    residual_sequence = images - median_frame

    residual_image = combine_residual_stack(
        residual_stack=residual_sequence,
        angles=angles,
        combine="mean")

    return residual_image


def combine_residual_stack(residual_stack,
                           angles,
                           combine="mean",
                           subtract_temporal_average=False,
                           num_cpus=4):

    if combine == "mean":
        temporal_average = np.mean(residual_stack, axis=0)
    else:
        temporal_average = np.median(residual_stack, axis=0)

    if subtract_temporal_average:
        residual_stack = residual_stack - temporal_average

    if num_cpus == 1:
        de_rotated = []

        for i in tqdm(range(residual_stack.shape[0])):
            tmp_frame = rotate(
                residual_stack[i, :, :],
                -np.rad2deg(angles[i]),
                (0, 1),
                False)
            de_rotated.append(tmp_frame)

    else:
        arguments = [(residual_stack[i, :, :],
                      -np.rad2deg(angles[i]),
                      (0, 1),
                      False) for i in range(residual_stack.shape[0])]

        pool = multiprocessing.Pool(num_cpus)
        de_rotated = np.array(pool.starmap(rotate, arguments))
        pool.close()

    if combine == "mean":
        return np.mean(de_rotated, axis=0)
    else:
        return np.median(de_rotated, axis=0)
