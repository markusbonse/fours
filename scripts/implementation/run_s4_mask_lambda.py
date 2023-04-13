import sys
import json
import numpy as np

from s4hci.utils.data_handling import load_adi_data
from s4hci.models.noise import S4Ridge
from s4hci.utils.logging import print_message, setup_logger

if __name__ == '__main__':
    setup_logger()
    print_message("Start Main")

    # 1.) Read in the setup
    dataset_config_file = str(sys.argv[1])
    lambda_reg = float(str(sys.argv[2]))
    convolve = bool(str(sys.argv[3]))
    cut_radius_psf = float(str(sys.argv[4]))
    mask_radius = float(str(sys.argv[5]))
    save_file = str(sys.argv[6])

    print_message("Parameters loaded")

    # 2.) Load the data
    with open(dataset_config_file) as json_file:
        dataset_config = json.load(json_file)

    print_message("Loading data ...")
    science_data, raw_angles, raw_psf_template_data = \
        load_adi_data(dataset_config["file_path"],
                      data_tag=dataset_config["stack_key"],
                      psf_template_tag=dataset_config["psf_template_key"],
                      para_tag=dataset_config["parang_key"])

    psf_template_data = np.mean(raw_psf_template_data, axis=0)

    X_train = science_data[0::2]
    X_test = science_data[1::2]

    # 3.) Create S4Ridge
    print_message("Creating S4Ridge")

    s4_ridge = S4Ridge(
        psf_template=psf_template_data,
        lambda_reg=lambda_reg,
        convolve=True,
        use_normalization=True,
        available_devices=[0],
        half_precision=False,
        cut_radius_psf=cut_radius_psf,
        mask_template_setup=("radius", mask_radius))

    # 4.) Fit the data
    print_message("Fit the data")
    s4_ridge.fit(science_data)

    # 5.) Save the result
    print_message("Save results")
    s4_ridge.save(save_file)

    print_message("Finished")
