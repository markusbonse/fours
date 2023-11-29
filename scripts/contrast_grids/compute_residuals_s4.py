import sys
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from applefy.detections.contrast import Contrast

from s4hci.utils.data_handling import load_adi_data
from s4hci.detection_limits.applefy_wrapper import S4DataReduction

from s4hci.utils.logging import print_message, setup_logger
from s4hci.utils.setups import contrast_grid_setup_1


if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_file = Path(str(sys.argv[1]))
    experiment_root_dir = Path(str(sys.argv[2]))
    exp_id = str(sys.argv[3])
    json_file = Path(str(sys.argv[4]))

    with open(json_file) as f:
        parameter_config = json.load(f)

    num_epochs = int(parameter_config["num_epochs"])
    dit_psf_template = float(parameter_config["dit_psf"])
    dit_science = float(parameter_config["dit_science"])
    fwhm = float(parameter_config["fwhm"])
    scaling_factor = float(parameter_config["nd_scaling"])

    # 2.) Load the dataset
    print_message("Loading dataset " + str(dataset_file))
    science_data, angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=str(dataset_file),
            data_tag="object_stacked_05",
            psf_template_tag="psf_template",
            para_tag="header_object_stacked_05/PARANG")

    psf_template = np.median(raw_psf_template_data, axis=0)

    # we cut the image to 91 x 91 pixel to be slightly larger than 1.2 arcsec
    cut_off = int((science_data.shape[1] - 91) / 2)
    science_data = science_data[:, cut_off:-cut_off, cut_off:-cut_off]

    # Create fake planet experiments
    print_message("Restore fake planet experiments")
    contrast_instance = Contrast(
        science_sequence=science_data,
        psf_template=psf_template,
        parang_rad=angles,
        psf_fwhm_radius=fwhm / 2,
        dit_psf_template=dit_psf_template,
        dit_science=dit_science,
        scaling_factor=scaling_factor,
        checkpoint_dir=experiment_root_dir)

    # get fake planet setup
    flux_ratios, separations, num_fake_planets = contrast_grid_setup_1(fwhm)

    # the config files exist already. We only need to create the setups in
    # the contrast instance. This can be done by setting the config_dir to
    # None and restoring it afterwards.
    tmp_config_dir = deepcopy(contrast_instance.config_dir)
    contrast_instance.config_dir = None
    contrast_instance.design_fake_planet_experiments(
        flux_ratios=flux_ratios,
        num_planets=num_fake_planets,
        separations=separations,
        overwrite=True)
    contrast_instance.config_dir = tmp_config_dir

    # 3.) Create S4 model
    print_message("Create S4 model")
    work_dir = contrast_instance.scratch_dir / \
        Path("tensorboard_S4")
    work_dir.mkdir(exist_ok=True)

    for lambda_reg, special_name in [
        (100, "lambda_000100"),
        (1000, "lambda_001000"),
        (10000, "lambda_010000"),
        (100000, "lambda_100000")]:

        print_message("Run fake planet experiments " + special_name)

        s4_model = S4DataReduction(
            device=0,
            lambda_reg=lambda_reg,
            special_name=special_name,
            rotation_grid_down_sample=1,
            logging_interval=50,
            save_models=True,
            convolve=True,
            train_num_epochs=num_epochs,
            noise_cut_radius_psf=fwhm,
            noise_mask_radius=fwhm * 1.5,
            work_dir=str(work_dir),
            verbose=True)

        # 5.) Run the fake planet experiments

        _ = contrast_instance._run_fake_planet_experiment(
            algorithm_function=s4_model,
            exp_id=exp_id)

    print_message("Finished Main")
