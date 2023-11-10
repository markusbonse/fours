import sys
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
    use_rotation_loss = bool(int(sys.argv[4]))
    num_epochs = int(sys.argv[5])
    lambda_reg = float(sys.argv[6])

    # 2.) Load the dataset
    print_message("Loading dataset " + str(dataset_file))
    science_data, angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=str(dataset_file),
            data_tag="object_stacked_05",
            psf_template_tag="psf_template",
            para_tag="header_object_stacked_05/PARANG")

    psf_template = np.median(raw_psf_template_data, axis=0)

    # other parameters
    dit_psf_template = 0.0042560
    dit_science = 0.08
    fwhm = 3.6
    pixel_scale = 0.02718

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
        scaling_factor=1.,
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
    if use_rotation_loss:
        special_name = "S4_rotation_loss"
    else:
        special_name = "S4_old_loss"

    work_dir = contrast_instance.scratch_dir / \
        Path("tensorboard_" + special_name)
    work_dir.mkdir(exist_ok=True)

    s4_model = S4DataReduction(
        device=0,
        special_name=special_name,
        work_dir=str(work_dir),
        verbose=True)

    s4_model.setup_create_noise_model(
        lambda_reg=lambda_reg,
        use_rotation_loss=use_rotation_loss,
        rotation_grid_down_sample=1,
        noise_cut_radius_psf=None,
        noise_mask_radius=None,
        logging_interval=5,
        noise_normalization="robust",
        convolve=True,
        train_num_epochs=num_epochs)

    # 5.) Run the fake planet experiments
    print_message("Run fake planet experiments")
    _ = contrast_instance._run_fake_planet_experiment(
        algorithm_function=s4_model,
        exp_id=exp_id)

    print_message("Finished Main")
