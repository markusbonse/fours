import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from applefy.detections.contrast import Contrast

from s4hci.utils.data_handling import load_adi_data
from s4hci.detection_limits.applefy_wrapper import S4DataReduction
from applefy.utils import mag2flux_ratio

from applefy.utils.positions import center_subpixel
from s4hci.utils.logging import print_message, setup_logger


if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_file = Path(str(sys.argv[1]))
    experiment_root_dir = Path(str(sys.argv[2]))
    exp_id = str(sys.argv[3])
    lambda_reg = 1289.89  # float(sys.argv[5])

    # 2.) Load the dataset
    print_message("Loading dataset " + str(dataset_file))
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=str(dataset_file),
            data_tag="object",
            psf_template_tag="psf_template",
            para_tag="header_object/PARANG")

    science_data = science_data[:, 12:-12, 12:-12]

    # Background subtraction of the PSF template
    psf_template = np.median(raw_psf_template_data, axis=0)
    psf_template = psf_template - np.min(psf_template)

    # other parameters
    dit_psf_template = 0.0042560
    dit_science = 0.08
    fwhm = 3.6

    # Create fake planet experiments
    print_message("Restore fake planet experiments")
    tmp_exp_root = experiment_root_dir / dataset_file.name[:-5]
    tmp_exp_root.mkdir(exist_ok=True)
    model_save_dir = tmp_exp_root / Path("models/" + exp_id + "/")
    model_save_dir.mkdir(exist_ok=True)

    contrast_instance = Contrast(
        science_sequence=science_data,
        psf_template=psf_template,
        parang_rad=raw_angles,
        psf_fwhm_radius=fwhm / 2,
        dit_psf_template=dit_psf_template,
        dit_science=dit_science,
        scaling_factor=1.,
        checkpoint_dir=tmp_exp_root)

    # fake planet brightness
    flux_ratios_mag = np.linspace(5., 13, 17)
    flux_ratios = mag2flux_ratio(flux_ratios_mag)

    center = center_subpixel(science_data[0])
    separations = np.arange(fwhm, fwhm * 6.5, fwhm / 2)[1:]
    num_fake_planets = 3

    # the config files exist already. We only need to create the setups in
    # the contrast instance
    tmp_config_dir = deepcopy(contrast_instance.config_dir)
    contrast_instance.config_dir = None
    contrast_instance.design_fake_planet_experiments(
        flux_ratios=flux_ratios,
        num_planets=num_fake_planets,
        separations=separations,
        overwrite=True)
    contrast_instance.config_dir = tmp_config_dir

    # 4.) Create S4 model
    extra_name = "fine_tune_robust"
    print_message("Create S4 model")
    s4_model = S4DataReduction(
        special_name="final_" + extra_name,
        noise_model_file=None,
        normalization_model_file=None,
        device=0,
        work_dir=str(model_save_dir) + "/" + extra_name + "/",
        verbose=True)

    # 4.1) Make sure we create the model from scratch
    s4_model.setup_learning_noise_model(
        noise_cut_radius_psf=4.0,
        noise_mask_radius=5.5,
        convolve=True,
        noise_normalization="robust",
        lambda_reg=lambda_reg,
        save_models=True,
        num_epochs_fine_tune_noise=200,
        learning_rate_fine_tune_noise=1e-6,
        batch_size_fine_tune_noise=-1)

    # 4.2) Make sure we build the planet model
    s4_model.setup_leaning_planet_model(
        num_epochs=500,
        create_raw_residuals=True,
        fine_tune_noise_model=True,
        save_models=False,
        learning_rate_planet=1e-3,
        learning_rate_noise=1e-6,
        batch_size=-1,
        rotation_grid_down_sample=1,
        upload_rotation_grid=True,
        logging_interval=50,
        planet_convolve_second=True,
        planet_use_up_sample=1)

    # 5.) Run the fake planet experiments
    print_message("Run fake planet experiments")
    _ = contrast_instance._run_fake_planet_experiment(
        algorithm_function=s4_model,
        exp_id=exp_id)

    print_message("Finished Main")
