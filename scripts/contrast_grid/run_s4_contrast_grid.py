import json
import sys
import numpy as np
from pathlib import Path

from s4hci.utils.data_handling import load_adi_data, save_as_fits
from s4hci.models.psf_subtraction import S4
from s4hci.models.noise import S4Noise
from s4hci.utils.logging import print_message, setup_logger

from applefy.utils.fake_planets import add_fake_planets

if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_hdf5_file = str(sys.argv[1])
    fake_planet_config_file = str(sys.argv[2])
    reg_lambda = float(sys.argv[3])
    s4_work_dir = str(sys.argv[4])

    # 2.) Load the dataset
    print_message("Loading dataset")
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=dataset_hdf5_file,
            data_tag="object",
            psf_template_tag="psf_template",
            para_tag="header_object/PARANG")

    psf_template_data = np.mean(raw_psf_template_data, axis=0)

    # Background subtraction of the PSF template
    psf_template_data = psf_template_data - np.min(psf_template_data)

    # 3.) Add the fake planet
    print_message("Add fake planet")
    with open(fake_planet_config_file) as json_file:
        fake_planet_config = json.load(json_file)

    data_with_fake_planet = add_fake_planets(
        input_stack=science_data,
        psf_template=psf_template_data,
        parang=raw_angles,
        dit_psf_template=0.004256,
        dit_science=0.08,
        experiment_config=fake_planet_config,
        scaling_factor=1.0)

    # 4.) Build the Raw model
    print_message("Find closed form solution")
    s4_model = S4(
        data_cube=data_with_fake_planet,
        parang=raw_angles,
        psf_template=psf_template_data,
        noise_noise_cut_radius_psf=4.0,
        noise_mask_radius=5.5,
        device=0,
        noise_lambda_init=reg_lambda,
        planet_convolve_second=True,
        planet_use_up_sample=1,
        work_dir=s4_work_dir,
        verbose=True)

    s4_model.find_closed_form_noise_model(save_model=True)

    # 5.) Create the residual
    print_message("Compute residuals")
    residual_before_fine_tuning = s4_model.compute_residual(
        account_for_planet=False)

    save_as_fits(
        residual_before_fine_tuning,
        s4_model.residuals_dir / Path(
            "01_Residual_Raw.fits"),
        overwrite=True)

    # 6.) Fine-tune the model (only planet model)
    print_message("Fine-tune model")
    s4_model.fine_tune_model_with_planet(
        200,
        learning_rate_planet=1e-3,
        learning_rate_noise=1e-6,
        fine_tune_noise_model=False,
        rotation_grid_down_sample=10,
        upload_rotation_grid=True,
        batch_size=10000)

    # save the models
    s4_model.noise_model.save(
        s4_model.models_dir / "noise_model_fine_tuned_only_planet.pkl")
    s4_model.planet_model.save(
        s4_model.models_dir / "planet_model_fine_tuned_only_planet.pkl")

    # 7.) Compute residuals
    print_message("Compute residuals")
    residual_with_planet_model = s4_model.compute_residual(
        account_for_planet=True)
    save_as_fits(
        residual_with_planet_model,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tune_planet_only_wp.fits"),
        overwrite=True)

    residual_no_planet_model = s4_model.compute_residual(
        account_for_planet=False)
    save_as_fits(
        residual_no_planet_model,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tune_planet_only_np.fits"),
        overwrite=True)

    # 8.) Restore the raw model
    print_message("Load previous model")
    save_path_final = s4_model.models_dir / "noise_model_closed_form.pkl"
    s4_noise = S4Noise.load(save_path_final)
    s4_model.noise_model = s4_noise

    # 9.) Fine-tune the model (with noise model)
    print_message("Fine-tune model")
    s4_model.fine_tune_model_with_planet(
        200,
        learning_rate_planet=1e-3,
        learning_rate_noise=1e-6,
        fine_tune_noise_model=True,
        rotation_grid_down_sample=10,
        upload_rotation_grid=True,
        batch_size=10000)

    # save the models
    s4_model.noise_model.save(
        s4_model.models_dir / "noise_model_fine_tuned.pkl")
    s4_model.planet_model.save(
        s4_model.models_dir / "planet_model_fine_tuned.pkl")

    # 10.) Compute residuals
    print_message("Compute residuals")
    residual_with_planet_model = s4_model.compute_residual(
        account_for_planet=True)

    save_as_fits(
        residual_with_planet_model,
        s4_model.residuals_dir / Path(
            "03_Residual_Fine_tune_wp.fits"),
        overwrite=True)

    residual_no_planet_model = s4_model.compute_residual(
        account_for_planet=False)
    save_as_fits(
        residual_no_planet_model,
        s4_model.residuals_dir / Path(
            "03_Residual_Fine_tune_np.fits"),
        overwrite=True)
