import json
import sys
import gc
import re
import numpy as np
from pathlib import Path

from s4hci.utils.data_handling import load_adi_data
from s4hci.models.noise import S4ClosedForm
from s4hci.utils.logging import print_message, setup_logger
from s4hci.utils.adi_tools import combine_residual_stack
from s4hci.utils.masks import construct_rfrr_mask

from applefy.utils.fake_planets import add_fake_planets
from applefy.utils.file_handling import save_as_fits


if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    fake_planet_config_file = str(sys.argv[1])
    mask_size = float(sys.argv[2])
    dataset_file = str(sys.argv[3])
    models_root_dir = Path(str(sys.argv[4]))
    residual_path = Path(str(sys.argv[5]))

    # 2.) Load the dataset
    print_message("Loading dataset")
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            dataset_file,
            data_tag="13_object_final",
            psf_template_tag="10_psf",
            para_tag="header_13_object_final/PARANG")

    psf_template_data = np.mean(raw_psf_template_data, axis=0)

    # 3.) add the fake planet
    print_message("Adding fake planet")
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

    # 4.) prepare the data
    X_train = data_with_fake_planet[0::2]
    angles_train = raw_angles[0::2]
    X_test = science_data[1::2]
    angles_test = raw_angles[1::2]

    # 5.) Clean up
    del science_data
    gc.collect()

    # 6.) Create the residuals
    print_message("Creating residuals")
    for tmp_model_file in models_root_dir.iterdir():
        if not tmp_model_file.name.endswith(".pkl"):
            continue

        # get the current parameters
        tmp_lambda = float(
            re.search("_lamb_(\d)+.(\d)+",
                      tmp_model_file.name).group(0)[6:])

        tmp_mask = float(
            re.search("_mask_(\d)+.(\d)+",
                      tmp_model_file.name).group(0)[6:])

        if tmp_mask != mask_size:
            print("mask size " + str(tmp_mask) + " does not match"
                  + str(mask_size))
            continue

        planet_name = fake_planet_config["exp_id"]
        dataset_name = planet_name + "_mask_" + str(tmp_mask) + "_lamb_" + str(
            tmp_lambda) + ".fits"
        residual_file_mean = str(residual_path / Path(
            "mean_residual_" + dataset_name))

        residual_file_median = str(residual_path / Path(
            "median_residual_" + dataset_name))

        if Path(residual_file_median).is_file() and \
                Path(residual_file_mean).is_file():
            print("residual already exists")
            continue

        print_message("Creating residual for " + str(tmp_lambda))

        # restore the model
        s4_ridge = S4ClosedForm.restore_from_checkpoint(
            checkpoint_file=tmp_model_file)

        # predict training data
        noise_model, residual = s4_ridge.predict(X_train)

        # predict test data
        noise_model_test, residual_test = s4_ridge.predict(X_test)

        # build error frame
        mean_error_frame = np.mean(np.abs(residual_test), axis=0)
        median_error_frame = np.median(np.abs(residual_test), axis=0)

        # build the residuals
        final_residual_mean = combine_residual_stack(
            residual_stack=np.array(residual),
            angles=angles_train,
            combine=["Mean_Residuals"],
            num_cpus=1)

        final_residual_median = combine_residual_stack(
            residual_stack=np.array(residual) - np.median(residual, axis=0),
            angles=angles_train,
            combine=["Median_Residuals"],
            num_cpus=1)

        # save the result
        print_message("Saving residual for " + str(tmp_lambda))

        if not Path(residual_file_mean).is_file():
            save_as_fits(
                final_residual_mean["Mean_Residuals"],
                residual_file_mean)

        if not Path(residual_file_median).is_file():
            save_as_fits(
                final_residual_median["Median_Residuals"],
                residual_file_median)

        if not (residual_path / Path("median_error_" + dataset_name)).is_file():
            save_as_fits(
                median_error_frame,
                str(residual_path / Path("median_error_" + dataset_name)))

        if not (residual_path / Path("mean_error_" + dataset_name)).is_file():
            save_as_fits(
                mean_error_frame,
                str(residual_path / Path("mean_error_" + dataset_name)))
