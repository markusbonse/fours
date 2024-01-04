import sys
import json
from copy import deepcopy
from applefy.detections.contrast import Contrast

from s4hci.utils.data_handling import load_adi_data
from s4hci.detection_limits.applefy_wrapper import S4DataReduction

from s4hci.utils.logging import print_message, setup_logger
from s4hci.utils.setups import contrast_grid_setup_1

from pathlib import Path
import numpy as np

# Methods
from s4hci.models.psf_subtraction import S4
from s4hci.utils.pca import pca_psf_subtraction_gpu

from s4hci.utils.data_handling import save_as_fits
from applefy.utils.file_handling import load_adi_data


if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the argument
    dataset_file = Path(str(sys.argv[1]))
    tmp_result_dir = Path("/fast/mbonse/NACO/70_results") / \
        dataset_file.name[:-5]
    tmp_result_dir.mkdir(exist_ok=True)
    work_dir = tmp_result_dir / Path("S4")
    work_dir.mkdir(exist_ok=True)

    fwhm = 3.6
    pixel_scale = 0.0271

    # 2.) Load the dataset
    science_data, angles, raw_psf_template_data = load_adi_data(
        str(dataset_file),
        data_tag="object_stacked_05",
        psf_template_tag="psf_template",
        para_tag="header_object_stacked_05/PARANG")

    # Background subtraction of the PSF template
    psf_template_data = np.median(raw_psf_template_data, axis=0)

    # 3.) Run the S4 model
    for lambda_reg, special_name in [
        (100, "lambda_000100"),
        (1000, "lambda_001000"),
        (10000, "lambda_010000"),
        (100000, "lambda_100000")]:

        print_message("Run fake planet experiments " + special_name)

        s4_model = S4(
            science_cube=science_data,
            adi_angles=angles,
            psf_template=psf_template_data,
            device=0,
            work_dir=work_dir,
            verbose=True,
            rotation_grid_subsample=1,
            noise_model_lambda_init=lambda_reg,
            noise_cut_radius_psf=fwhm,
            noise_mask_radius=fwhm * 1.5,
            noise_normalization="normal",
            noise_model_convolve=True)

        s4_model.fit_noise_model(
            num_epochs=500,
            use_rotation_loss=True,
            training_name=special_name,
            logging_interval=1)

    print_message("Finished Main")
