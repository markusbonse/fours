import numpy as np
import multiprocessing
from scipy.ndimage import rotate


def combine_residual_stack(residual_stack,
                           angles,
                           combine,
                           suffix="",
                           num_cpus=4):

    arguments = [(residual_stack[i, :, :],
                  -np.rad2deg(angles[i]),
                  (0, 1),
                  False) for i in range(residual_stack.shape[0])]

    pool = multiprocessing.Pool(num_cpus)
    de_rotated = np.array(pool.starmap(rotate, arguments))
    pool.close()

    results = dict()
    if "Mean_Residuals" in combine:
        results["Mean_Residuals" + suffix] = np.mean(de_rotated, axis=0)

    if "MedianResiduals" in combine:
        results["Median_Residuals" + suffix] = np.median(de_rotated, axis=0)

    if "Mean_Residuals_NoiseNorm" in combine:
        results["Mean_Residuals_NoiseNorm" + suffix] = np.divide(
            np.mean(de_rotated, axis=0),
            np.std(de_rotated, axis=0),
            out=np.zeros_like(np.mean(de_rotated, axis=0)),
            where=np.std(de_rotated, axis=0) != 0)

    return results
