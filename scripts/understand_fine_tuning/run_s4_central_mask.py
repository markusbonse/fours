import sys
import pickle
import numpy as np
from pathlib import Path

from s4hci.utils.data_handling import load_adi_data, save_as_fits
from s4hci.models.psf_subtraction import S4
from s4hci.utils.logging import print_message, setup_logger
from s4hci.utils.masks import construct_central_mask


if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_hdf5_file = str(sys.argv[1])
    s4_work_dir = str(sys.argv[2])

    print(dataset_hdf5_file)
    Path(s4_work_dir).mkdir(exist_ok=True, parents=True)

    # 2.) Load the dataset
    print_message("Loading dataset")
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=dataset_hdf5_file,
            data_tag="object",
            psf_template_tag="psf_template",
            para_tag="header_object/PARANG")

    science_data = science_data[:, 17:-17, 17:-17]

    # 2.1) Split the data into training and test data
    # Note: The test data is used later in the notebooks
    science_data_train = science_data[0::2]
    angles_train = raw_angles[0::2]

    # apply the central mask
    mask = construct_central_mask(science_data.shape[-1], 6 / 2)
    science_data_train = science_data_train * mask

    # Background subtraction of the PSF template
    psf_template_data = np.median(raw_psf_template_data, axis=0)
    psf_template_data = psf_template_data - np.min(psf_template_data)

    # 3.) Build the Raw model
    print_message("Build S4 model")
    s4_model = S4(
        data_cube=science_data_train,
        parang=angles_train,
        psf_template=psf_template_data,
        noise_noise_cut_radius_psf=4.0,
        noise_mask_radius=5.5,
        device=0,
        convolve=True,
        noise_normalization="normal",
        planet_convolve_second=True,
        planet_use_up_sample=1,
        work_dir=s4_work_dir,
        verbose=True)

    # 4.) Run the validation to find the best lambda value
    print_message("Find best lambda value")
    lambdas = np.logspace(1, 8, 200)
    validation_results, _ = s4_model.validate_lambdas_noise(
        num_separations=20,
        lambdas=lambdas,
        num_test_positions=10,
        test_size=0.3,
        approx_svd=5000)

    validation_save_file = s4_model.models_dir / Path("validation_results.pkl")

    with open(validation_save_file, 'wb') as handle:
        pickle.dump(validation_results,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # 5.) Find the closed form solution
    print_message("Find closed form solution")
    s4_model.find_closed_form_noise_model(
        save_model=True,
        lstrq_mode="LSTSQ",
        fp_precision="float32")

    # 6.) Compute residuals
    print_message("Compute residuals median")
    residual_median_raw = s4_model.compute_residual(
        account_for_planet=False,
        combine="median")

    save_as_fits(
        residual_median_raw,
        s4_model.residuals_dir / Path(
            "01_Residual_Raw_median.fits"),
        overwrite=True)

    # 8.) Compute residuals - mean
    print_message("Compute residuals mean")
    residual_mean_raw = s4_model.compute_residual(
        account_for_planet=False,
        combine="mean")

    save_as_fits(
        residual_mean_raw,
        s4_model.residuals_dir / Path(
            "01_Residual_Raw_mean.fits"),
        overwrite=True)

    # 7.) Fine-tune the model (only planet model)
    print_message("Fine-tune model")
    test_model_file = s4_model.models_dir / \
        "noise_model_fine_tuned_only_planet.pkl"

    s4_model.fine_tune_model(
        num_epochs=500,
        learning_rate_planet=1e-3,
        learning_rate_noise=1e-6,
        fine_tune_noise_model=True,
        lean_planet_model=False,
        rotation_grid_down_sample=10,
        upload_rotation_grid=True,
        batch_size=-1)

    # save the models
    s4_model.noise_model.save(
        s4_model.models_dir / "noise_model_fine_tuned.pkl")

    # 8.) Compute residuals - median
    print_message("Compute residuals median")
    residual_median_fine_tuned = s4_model.compute_residual(
        account_for_planet=False,
        combine="median")

    save_as_fits(
        residual_median_fine_tuned,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tuned_median.fits"),
        overwrite=True)

    # 8.) Compute residuals - mean
    print_message("Compute residuals mean")
    residual_mean_fine_tuned = s4_model.compute_residual(
        account_for_planet=False,
        combine="mean")

    save_as_fits(
        residual_mean_fine_tuned,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tuned_mean.fits"),
        overwrite=True)

    print_message("Finished Main")
