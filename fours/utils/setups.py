from typing import Tuple

import numpy as np
from applefy.utils import mag2flux_ratio


def contrast_grid_setup_1(
        fwhm: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    This is a simple setup function which creates the input parameters for the
    contrast grids used in the paper. Usually this function is not needed
    unless you want to reproduce the results of the paper.

    Parameters:
    -----------
    fwhm : float
        The full width at half maximum (FWHM) used to determine separations.

    Returns:
    --------
    tuple
        flux_ratios : numpy.ndarray
            Array of flux ratios.
        separations : numpy.ndarray
            Array of separations in terms of multiples of FWHM.
        num_fake_planets : int
            The number of fake planets to simulate in the grid.
    """
    

    flux_ratios_mag = np.linspace(5., 15, 21)
    flux_ratios = mag2flux_ratio(flux_ratios_mag)
    separations = np.concatenate([
        np.arange(fwhm, fwhm * 5.5, fwhm / 2)[1:],
        np.arange(fwhm * 6, fwhm * 12.5, fwhm)])

    num_fake_planets = 3

    return flux_ratios, separations, num_fake_planets
