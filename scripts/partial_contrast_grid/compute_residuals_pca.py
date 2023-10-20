import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from applefy.detections.contrast import Contrast

from s4hci.utils.data_handling import load_adi_data
from s4hci.detection_limits.applefy_wrapper import PCADataReductionGPU
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
    print_message("Create PCA model")
    pca_numbers = np.concatenate(
            [np.arange(0, 20, 2)[1:],
             np.arange(20, 50, 5),
             np.arange(50, 100, 10),
             np.arange(100, 200, 20),
             np.arange(200, 550, 50)])

    pca_model = PCADataReductionGPU(
        approx_svd=10000,
        pca_numbers=pca_numbers,
        device=0,
        verbose=True)

    # 5.) Run the fake planet experiments
    print_message("Run fake planet experiments")
    _ = contrast_instance._run_fake_planet_experiment(
        algorithm_function=pca_model,
        exp_id=exp_id)

    print_message("Finished Main")
