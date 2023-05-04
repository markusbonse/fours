import sys
import json
import numpy as np
from pathlib import Path

from s4hci.utils.data_handling import load_adi_data
from s4hci.models.noise import S4ClosedForm
from s4hci.utils.logging import print_message, setup_logger

if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Load the config file
    tmp_config_file = str(sys.argv[1])
    print_message("Current config file is: " + tmp_config_file)

    with open(tmp_config_file) as json_file:
        config_data = json.load(json_file)

    # 2.) Parse all the information we need
    dataset_path = config_data["dataset_path"]
    data_tag = config_data["data_tag"]
    psf_template_tag = config_data["psf_template_tag"]
    para_tag = config_data["para_tag"]

    lambda_reg = config_data["lambda_reg"]
    mask_radius = config_data["mask_size"]
    cut_radius_psf = config_data["cut_radius_psf"]
    convolve = config_data["convolve"]
    use_normalization = config_data["use_normalization"]
    save_file = config_data["save_file"]

    print_message("Parameters loaded")
    if Path(save_file).is_file():
        sys.exit()

    # 2.) Load the data
    print_message("Loading data ...")
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(
            dataset_path,
            data_tag=data_tag,
            psf_template_tag=psf_template_tag,
            para_tag=para_tag)

    psf_template_data = np.mean(raw_psf_template_data, axis=0)

    X_train = science_data[0::2]
    X_test = science_data[1::2]

    # 3.) Create S4ClosedForm
    print_message("Creating S4ClosedForm")

    s4_ridge = S4ClosedForm(
        psf_template=psf_template_data,
        lambda_reg=lambda_reg,
        convolve=True,
        use_normalization=use_normalization,
        available_devices=[0],
        half_precision=False,
        cut_radius_psf=cut_radius_psf,
        mask_template_setup=("radius", mask_radius))

    # 4.) Fit the data
    print_message("Fit the data")
    s4_ridge.fit(X_train)
    #s4_ridge._setup_training(science_data)
    #s4_ridge.science_data_norm = s4_ridge.normalize_data(science_data)

    #positions = [(y, x)
    #             for x in range(0, s4_ridge.image_size, 20)
    #             for y in range(0, s4_ridge.image_size, 20)]

    #s4_ridge.betas = s4_ridge._fit_mp(positions)

    # 5.) Save the result
    print_message("Save results")
    s4_ridge.save(save_file)

    print_message("Finished")
