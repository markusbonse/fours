import numpy as np
from applefy.utils import mag2flux_ratio


def contrast_grid_setup_1(fwhm):

    flux_ratios_mag = np.linspace(5., 15, 21)
    flux_ratios = mag2flux_ratio(flux_ratios_mag)
    separations = np.concatenate([
        np.arange(fwhm, fwhm * 5.5, fwhm / 2)[1:],
        np.arange(fwhm * 6, fwhm * 12.5, fwhm)])

    num_fake_planets = 3

    return flux_ratios, separations, num_fake_planets
