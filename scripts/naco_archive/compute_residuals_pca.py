import sys

from s4hci.utils.logging import print_message, setup_logger

from pathlib import Path
import numpy as np

# Methods
from s4hci.utils.pca import pca_psf_subtraction_gpu

from s4hci.utils.data_handling import save_as_fits
from applefy.utils.file_handling import load_adi_data


if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the argument
    dataset_file = Path(str(sys.argv[1]))
    tmp_result_dir = Path("/fast/mbonse/NACO/70_results") / \
        dataset_file.name[:-9]
    tmp_result_dir.mkdir(exist_ok=True)
    work_dir = tmp_result_dir / Path("PCA")
    work_dir.mkdir(exist_ok=True)

    fwhm = 3.6
    pixel_scale = 0.0271

    # 2.) Load the dataset
    science_data, angles, raw_psf_template_data = load_adi_data(
        str(dataset_file),
        data_tag="object_hr_stacked_05",
        psf_template_tag="psf_template",
        para_tag="header_object_hr_stacked_05/PARANG")

    # Background subtraction of the PSF template
    psf_template_data = np.median(raw_psf_template_data, axis=0)

    # 3.) Run the PCA
    mean_residual = pca_psf_subtraction_gpu(
        images=science_data,
        angles=angles,
        pca_numbers=np.arange(500),
        device=0,
        approx_svd=np.min([science_data.shape[0] - 200, 5000]),
        subsample_rotation_grid=1,
        verbose=True,
        combine="mean")

    median_residual = pca_psf_subtraction_gpu(
        images=science_data,
        angles=angles,
        pca_numbers=np.arange(500),
        device=0,
        approx_svd=np.min([science_data.shape[0] - 200, 5000]),
        subsample_rotation_grid=1,
        verbose=True,
        combine="median")

    all_residuals = np.array([mean_residual, median_residual])
    save_as_fits(all_residuals,
                 str(work_dir / Path("pca_residuals.fits")),
                 overwrite=True)

    print_message("Finished Main")
