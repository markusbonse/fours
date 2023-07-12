import numpy as np
import h5py

from astropy.io import fits


def load_adi_data(hdf5_dataset: str,
                  data_tag: str,
                  psf_template_tag: str,
                  para_tag="PARANG"):
    """
    Function to load ADI data from hdf5 files
    :param hdf5_dataset: The path to the hdf5 file
    :param data_tag: Tag of the science data
    :param psf_template_tag: Tag of the PSF template
    :param para_tag: Tag of the parallactic angles
    :return: Tuple (Science, parang, template)
    """
    hdf5_file = h5py.File(hdf5_dataset, 'r')
    data = hdf5_file[data_tag][...]
    angles = np.deg2rad(hdf5_file[para_tag][...])
    psf_template_data = hdf5_file[psf_template_tag][...]
    hdf5_file.close()

    return data, angles, psf_template_data


def save_as_fits(
        data: np.ndarray,
        file_name: str,
        overwrite:bool
) -> None:
    """
    Saves data as .fits file.

    Args:
        data: The data to be saved.
        file_name: The filename of the fits file.
    """

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])

    hdul.writeto(file_name, overwrite=overwrite)
