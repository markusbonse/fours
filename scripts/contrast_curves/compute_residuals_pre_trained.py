import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from astropy.modeling.functional_models import Moffat2D
from astropy.modeling import fitting
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
    pre_trained_noise_model = str(sys.argv[3])
    pre_trained_normalization = str(sys.argv[4])
    exp_id = str(sys.argv[5])

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

    # 3.) Fit FWHM
    print_message("Fit FWHM")
    # Fit the data using astropy.modeling
    p_init = Moffat2D(amplitude=1e4, x_0=10, y_0=10)
    fit_p = fitting.LevMarLSQFitter()

    y, x = np.mgrid[:psf_template.shape[0], :psf_template.shape[1]]
    p = fit_p(p_init, x, y, psf_template)
    fwhm = np.round(p.fwhm, 1)

    # Create fake planet experiments
    print_message("Create fake planet experiments")
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
    flux_ratio_mag = 14
    flux_ratio = mag2flux_ratio(flux_ratio_mag)

    print_message("Brightness of fake planets in mag: " + str(flux_ratio_mag))
    print_message("Planet-to-star flux ratio: " + str(flux_ratio))

    center = center_subpixel(science_data[0])
    separations = np.arange(0, center[0], fwhm / 2.)[2:]
    num_fake_planets = 6

    if exp_id == "0000":
        contrast_instance.design_fake_planet_experiments(
            flux_ratios=flux_ratio,
            num_planets=num_fake_planets,
            separations=separations,
            overwrite=True)
    else:
        tmp_config_dir = deepcopy(contrast_instance.config_dir)
        contrast_instance.config_dir = None
        contrast_instance.design_fake_planet_experiments(
            flux_ratios=flux_ratio,
            num_planets=num_fake_planets,
            separations=separations,
            overwrite=True)
        contrast_instance.config_dir = tmp_config_dir

    # 4.) Create S4 model
    print_message("Create S4 model")
    s4_model = S4DataReduction(
        noise_model_file=pre_trained_noise_model,
        normalization_model_file=pre_trained_normalization,
        device="cpu",
        work_dir=None,
        verbose=True)

    # 5.) Run the fake planet experiments
    print_message("Run fake planet experiments")
    _ = contrast_instance._run_fake_planet_experiment(
        algorithm_function=s4_model,
        exp_id=exp_id)

    print_message("Finished Main")