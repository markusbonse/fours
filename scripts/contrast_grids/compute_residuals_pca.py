import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from applefy.detections.contrast import Contrast

from s4hci.utils.data_handling import load_adi_data
from s4hci.detection_limits.applefy_wrapper import PCADataReductionGPU

from s4hci.utils.logging import print_message, setup_logger
from s4hci.utils.setups import contrast_grid_setup_1


if __name__ == "__main__":
    setup_logger()
    print_message("Start Main")

    # 1.) Load the arguments
    dataset_file = Path(str(sys.argv[1]))
    experiment_root_dir = Path(str(sys.argv[2]))
    exp_id = str(sys.argv[3])
    use_stacking = bool(int(sys.argv[4]))

    # 2.) Load the dataset
    print_message("Loading dataset " + str(dataset_file))
    if use_stacking:
        data_tag = "object_stacked_05"
        pca_special_name = "pca_stacked_05"
    else:
        data_tag = "object"
        pca_special_name = "pca_non_stacked"

    science_data, angles, raw_psf_template_data = \
        load_adi_data(
            hdf5_dataset=str(dataset_file),
            data_tag=data_tag,
            psf_template_tag="psf_template",
            para_tag="header_" + data_tag + "/PARANG")

    psf_template = np.median(raw_psf_template_data, axis=0)

    # other parameters
    dit_psf_template = 0.0042560
    dit_science = 0.08
    fwhm = 3.6
    pixel_scale = 0.02718

    # we cut the image to 91 x 91 pixel to be slightly larger than 1.2 arcsec
    cut_off = int((science_data.shape[1] - 91) / 2)
    science_data = science_data[:, cut_off:-cut_off, cut_off:-cut_off]

    # 3.) Create fake planet experiments
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

    # 4.) Create S4 model
    print_message("Create PCA model")
    pca_numbers = np.concatenate(
            [np.arange(0, 20, 2)[1:],
             np.arange(20, 50, 5),
             np.arange(50, 100, 10),
             np.arange(100, 200, 20),
             np.arange(200, 550, 50)])

    # create the PCA logging directory
    work_dir = contrast_instance.scratch_dir / \
        Path("tensorboard_" + pca_special_name)

    pca_model = PCADataReductionGPU(
        approx_svd=8000,
        pca_numbers=pca_numbers,
        device=0,
        work_dir=work_dir,
        special_name=pca_special_name,
        verbose=True)

    # 5.) Run the fake planet experiments
    print_message("Run fake planet experiments")
    _ = contrast_instance._run_fake_planet_experiment(
        algorithm_function=pca_model,
        exp_id=exp_id)

    print_message("Finished Main")
