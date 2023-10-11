import sys
import pickle
import numpy as np
from pathlib import Path

from s4hci.utils.data_handling import load_adi_data, save_as_fits
from s4hci.models.psf_subtraction import S4
from s4hci.utils.logging import print_message, setup_logger


if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_hdf5_file = str(sys.argv[1])
    s4_work_dir = str(sys.argv[2])
    Path(s4_work_dir).mkdir(exist_ok=True, parents=True)

    # 2.) Load the dataset
    print_message("Loading dataset " + str(dataset_hdf5_file))
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=dataset_hdf5_file,
            data_tag="object",
            psf_template_tag="psf_template",
            para_tag="header_object/PARANG")

    science_data = science_data[:, 12:-12, 12:-12]

    # Background subtraction of the PSF template
    psf_template_data = np.median(raw_psf_template_data, axis=0)
    psf_template_data = psf_template_data - np.min(psf_template_data)

    # 3.) Build the Raw model
    print_message("Build S4 model")
    s4_model = S4(
        science_data=science_data,
        parang=raw_angles,
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
    if not (s4_model.models_dir / Path("validation_results.pkl")).is_file():
        print_message("Find best lambda value")
        lambdas = np.logspace(1, 8, 200)
        validation_results, _ = s4_model.validate_lambdas_noise(
            num_separations=20,
            lambdas=lambdas,
            num_test_positions=10,
            test_size=0.3,
            approx_svd=-1)

        validation_save_file = s4_model.models_dir / Path("validation_results.pkl")

        with open(validation_save_file, 'wb') as handle:
            pickle.dump(validation_results,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # 5.) Find the closed form solution
    if (s4_model.models_dir / Path("noise_model_raw.pkl")).is_file():
        print_message("Restore closed form solution")
        s4_model.restore_models(
            file_noise_model=
                s4_model.models_dir / Path("noise_model_raw.pkl"),
            file_normalization_model=
                s4_model.models_dir / Path("normalization_model.pkl"))
    else:
        print_message("Find closed form solution")
        s4_model.find_closed_form_noise_model(
            fp_precision="float32")

        # save the model
        s4_model.save_noise_model("noise_model_raw.pkl")
        s4_model.save_normalization_model("normalization_model.pkl")

    # 6.) Compute residuals
    print_message("Compute residuals median")
    residual_median_raw = s4_model.compute_residual(
        account_for_planet_model=False,
        combine="median")

    save_as_fits(
        residual_median_raw,
        s4_model.residuals_dir / Path(
            "01_Residual_Raw_median.fits"),
        overwrite=True)

    print_message("Compute residuals mean")
    residual_mean_raw = s4_model.compute_residual(
        account_for_planet_model=False,
        combine="mean")

    save_as_fits(
        residual_mean_raw,
        s4_model.residuals_dir / Path(
            "01_Residual_Raw_mean.fits"),
        overwrite=True)

    # 7.) Fine-tune the noise model
    s4_model.fine_tune_noise_model(
        num_epochs=500,
        learning_rate=1e-6,
        batch_size=-1)

    # save the model
    s4_model.save_noise_model("noise_model_fine_tuned.pkl")

    # 8.) Compute residuals
    print_message("Compute residuals median")
    residual_median_fine_tuned = s4_model.compute_residual(
        account_for_planet_model=False,
        combine="median")

    save_as_fits(
        residual_median_fine_tuned,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tuned_median.fits"),
        overwrite=True)

    # 8.) Compute residuals - mean
    print_message("Compute residuals mean")
    residual_mean_fine_tuned = s4_model.compute_residual(
        account_for_planet_model=False,
        combine="mean")

    save_as_fits(
        residual_mean_fine_tuned,
        s4_model.residuals_dir / Path(
            "02_Residual_Fine_tuned_mean.fits"),
        overwrite=True)

    print_message("Finished Main")
