import numpy as np

from astropy.modeling.functional_models import Moffat2D
from astropy.modeling import fitting


def get_fwhm(psf_template):
    # Fit the data using astropy.modeling
    p_init = Moffat2D(
        amplitude=np.max(psf_template),
        x_0=psf_template.shape[0] / 2,
        y_0=psf_template.shape[1] / 2)
    fit_p = fitting.LevMarLSQFitter()

    y, x = np.mgrid[:psf_template.shape[0],
           :psf_template.shape[1]]
    p = fit_p(p_init, x, y, psf_template)
    fwhm = np.round(p.fwhm, 1)

    return fwhm
