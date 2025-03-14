import os
import numpy as np
import h5py
from typing import Tuple

from astropy.io import fits


def load_adi_data(
        hdf5_dataset: str,
        data_tag: str,
        psf_template_tag: str,
        para_tag: str = "PARANG"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads Angular Differential Imaging (ADI) data from an HDF5 dataset. 
    See "Examples" in the Documentation.

    Args:
        hdf5_dataset: Path to the HDF5 file containing the data.
        data_tag: HDF5 key for the science data cube (e.g., images at 
            multiple wavelengths or observations).
        psf_template_tag: HDF5 key for the PSF template data.
        para_tag: HDF5 key for the parallactic angles array. Default is 
            "PARANG".

    Returns:
        1. A numpy array representing the science data cube.
        2. A numpy array of parallactic angles in radians.
        3. A numpy array representing the PSF template.
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
        overwrite:bool = False
) -> None:
    """
    Saves data as .fits file.

    Args:
        data: The data to be saved.
        file_name: The filename of the fits file.
        overwrite: Overwrite existing files
    """

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])

    hdul.writeto(file_name, overwrite=overwrite)


def read_fours_root_dir() -> str:
    """
    Reads the root directory for the FOURS dataset from an environment variable.

    The function checks if the `FOURS_ROOT_DIR` environment variable is correctly 
    set to a valid directory path, prints its location, and returns it. Raises an 
    IOError if the directory does not exist.

    Returns:
        A string representing the path to the FOURS_ROOT_DIR.

    Raises:
        IOError: If the directory specified in the `FOURS_ROOT_DIR` environment 
        variable does not exist.
    """

    root_dir = os.getenv('FOURS_ROOT_DIR')

    if not os.path.isdir(root_dir):
        raise IOError("The path in FOURS_ROOT_DIR does not exist. Make sure "
                      "to download the data and specify its location.")

    print("Data in the FOURS_ROOT_DIR found. Location: " + str(root_dir))
    return root_dir
