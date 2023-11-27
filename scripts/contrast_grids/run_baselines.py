from pathlib import Path
import sys
import json
from copy import deepcopy
import numpy as np

from applefy.detections.contrast import Contrast
from applefy.utils.file_handling import load_adi_data

from s4hci.utils.setups import contrast_grid_setup_1
from s4hci.utils.logging import print_message, setup_logger
from s4hci.detection_limits.applefy_wrapper import cADIDataReductionGPU, \
    PCADataReductionGPU

if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_file = Path(str(sys.argv[1]))
    experiment_root_dir = Path(str(sys.argv[2]))
    json_file = Path(str(sys.argv[3]))

    with open(json_file) as f:
        parameter_config = json.load(f)

    dit_psf_template = float(parameter_config["dit_psf"])
    dit_science = float(parameter_config["dit_science"])
    fwhm = float(parameter_config["fwhm"])
    scaling_factor = float(parameter_config["nd_scaling"])
    svd_approx = int(parameter_config["svd_approx"])

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

    # 1. Run fake planet experiments for cADI
    print_message("Run fake planet experiments for cADI")
    cadi_algorithm_function = cADIDataReductionGPU(0)
    contrast_instance.run_fake_planet_experiments(
        algorithm_function=cadi_algorithm_function,
        num_parallel=8)

    # 2. Run fake planet experiments for PCA
    print_message("Run fake planet experiments for PCA")
    pca_numbers = np.concatenate(
        [np.arange(0, 30, 2)[1:],
         np.arange(30, 100, 10),
         np.arange(100, 200, 20),
         np.arange(200, 550, 50)])

    work_dir = contrast_instance.scratch_dir / Path("tensorboard_pca")
    pca_algorithm_function = PCADataReductionGPU(
        approx_svd=svd_approx,
        pca_numbers=pca_numbers,
        device=0,
        work_dir=work_dir,
        special_name="stacked_05",
        verbose=False)

    old_results = contrast_instance.results_dict
    contrast_instance.run_fake_planet_experiments(
        algorithm_function=pca_algorithm_function,
        num_parallel=8)
    contrast_instance.results_dict.update(old_results)

    # finished
    print_message("Finished Main")
