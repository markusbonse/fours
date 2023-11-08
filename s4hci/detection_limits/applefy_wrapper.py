from typing import List, Dict, Union
import numpy as np
from applefy.detections.contrast import DataReductionInterface

from s4hci.models.psf_subtraction import S4
from s4hci.utils.pca import pca_psf_subtraction_gpu
from s4hci.utils.adi_tools import cadi_psf_subtraction


class cADIDataReduction(DataReductionInterface):

    def get_method_keys(self) -> List[str]:
        return ["cADI", ]

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str):

        residual = cadi_psf_subtraction(
            images=stack_with_fake_planet,
            angles=parang_rad)

        result_dict = dict()
        result_dict["cADI"] = residual

        return result_dict


class PCADataReductionGPU(DataReductionInterface):

    def __init__(
            self,
            pca_numbers: np.ndarray,
            approx_svd: int,
            device: Union[int, str] = "cpu",
            verbose: bool = False):
        self.pca_numbers = pca_numbers
        self.approx_svd = approx_svd
        self.device = device
        self.verbose = verbose

    def get_method_keys(self) -> List[str]:

        keys = ["PCA_" + str(num_pcas).zfill(3) + "_components"
                for num_pcas in self.pca_numbers]

        return keys

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:

        pca_residuals = pca_psf_subtraction_gpu(
            images=stack_with_fake_planet,
            angles=parang_rad,
            pca_numbers=self.pca_numbers,
            device=self.device,
            approx_svd=self.approx_svd,
            verbose=self.verbose)

        result_dict = dict()
        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = pca_residuals[idx]

        return result_dict


class S4DataReduction(DataReductionInterface):
    def __init__(
            self,
            special_name: str = None,
            noise_model_file: str = None,
            normalization_model_file: str = None,
            device: Union[int, str] = "cpu",
            work_dir: str = None,
            verbose: bool = False):

        self.special_name = special_name
        self.normalization_model_file = normalization_model_file
        self.noise_model_file = noise_model_file
        self.device = device
        self.work_dir = work_dir
        self.verbose = verbose
        self.planet_convolve_second = True
        self.planet_use_up_sample = 1

        if self.noise_model_file is not None \
                and self.normalization_model_file is not None:
            self.restore_possible = True
        else:
            self.restore_possible = False

        # 1.) will be set in setup_learning_noise_model
        self.noise_cut_radius_psf = None
        self.noise_mask_radius = None
        self.convolve = None
        self.noise_normalization = None
        self.lambda_reg = None

        # parameters for fine-tuning the noise model
        self.num_epochs_fine_tune_noise = None
        self.learning_rate_fine_tune_noise = None
        self.batch_size_fine_tune_noise = None
        self.lean_noise_model = False
        self.save_model_after_lean_noise_model = None

        # 2.) will be set in setup_leaning_planet_model
        self.create_raw_residuals = False
        self.num_epochs_learn_planet = None
        self.learning_rate_planet = None
        self.learning_rate_noise_learn_planet = None
        self.rotation_grid_down_sample = None
        self.upload_rotation_grid = None
        self.logging_interval = None
        self.fine_tune_noise_model_learn_planet = None
        self.batch_size_learn_planet = None

        self.save_model_after_lean_planet = None
        self.fine_tune_planet = False

        # will be created once the data is available
        self.s4_model = None

    def setup_learning_noise_model(
            self,
            noise_cut_radius_psf: float,
            noise_mask_radius: float,
            convolve: bool,
            noise_normalization: str,
            lambda_reg: float,
            save_models: bool = True,
            num_epochs_fine_tune_noise: int = 0,
            learning_rate_fine_tune_noise: float = 1e-6,
            batch_size_fine_tune_noise: int = -1):

        self.noise_cut_radius_psf = noise_cut_radius_psf
        self.noise_mask_radius = noise_mask_radius
        self.convolve = convolve
        self.noise_normalization = noise_normalization
        self.lambda_reg = lambda_reg
        self.num_epochs_fine_tune_noise = num_epochs_fine_tune_noise
        self.learning_rate_fine_tune_noise = learning_rate_fine_tune_noise
        self.batch_size_fine_tune_noise = batch_size_fine_tune_noise
        self.save_model_after_lean_noise_model = save_models
        self.lean_noise_model = True

    def setup_leaning_planet_model(
            self,
            num_epochs: int,
            fine_tune_noise_model: bool,
            create_raw_residuals: bool = False,
            save_models: bool = True,
            learning_rate_planet: float = 1e-3,
            learning_rate_noise: float = 1e-6,
            batch_size: int = -1,
            rotation_grid_down_sample: int = 1,
            upload_rotation_grid: bool = False,
            logging_interval: int = 10,
            planet_convolve_second: bool = True,
            planet_use_up_sample: int = 1):

        self.create_raw_residuals = create_raw_residuals
        self.planet_convolve_second = planet_convolve_second
        self.planet_use_up_sample = planet_use_up_sample
        self.num_epochs_learn_planet = num_epochs
        self.fine_tune_noise_model_learn_planet = fine_tune_noise_model
        self.learning_rate_planet = learning_rate_planet
        self.learning_rate_noise_learn_planet = learning_rate_noise
        self.rotation_grid_down_sample = rotation_grid_down_sample
        self.upload_rotation_grid = upload_rotation_grid
        self.logging_interval = logging_interval
        self.batch_size_learn_planet = batch_size
        self.save_model_after_lean_planet = save_models
        self.fine_tune_planet = True

    def get_method_keys(self) -> List[str]:
        if self.special_name is None:
            return ["s4_mean", "s4_median"]
        else:
            return ["s4_mean_" + self.special_name,
                    "s4_median_" + self.special_name]

    def _build_s4_noise_model(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray):

        # 1.) Create the S4 model
        self.s4_model = S4(
            science_cube=stack_with_fake_planet,
            adi_angles=parang_rad,
            psf_template=psf_template,
            noise_cut_radius_psf=self.noise_cut_radius_psf,
            noise_mask_radius=self.noise_mask_radius,
            device=self.device,
            noise_normalization=self.noise_normalization,
            noise_model_lambda_init=self.lambda_reg,
            planet_convolve_second=True,
            planet_use_up_sample=1,
            work_dir=self.work_dir,
            noise_model_convolve=self.convolve,
            verbose=self.verbose)

        # 2.) Find the closed form solution
        self.s4_model.find_closed_form_noise_model(fp_precision="float32")

        # 3.) Fine-tune the noise model
        if not self.num_epochs_fine_tune_noise == 0:
            self.s4_model.fine_tune_noise_model(
                num_epochs=self.num_epochs_fine_tune_noise,
                learning_rate=self.learning_rate_fine_tune_noise,
                batch_size=self.batch_size_fine_tune_noise)

        # 4.) Save the model if desired
        if self.save_model_after_lean_noise_model:
            self.s4_model.save_noise_model("noise_model_raw.pkl")
            self.s4_model.save_normalization_model("normalization_model.pkl")

    def _learn_planet_model(self):
        self.s4_model.learn_planet_model(
            num_epochs=self.num_epochs_learn_planet,
            learning_rate_planet=self.learning_rate_planet,
            learning_rate_noise=self.learning_rate_noise_learn_planet,
            fine_tune_noise_model=self.fine_tune_noise_model_learn_planet,
            rotation_grid_down_sample=self.rotation_grid_down_sample,
            upload_rotation_grid=self.upload_rotation_grid,
            logging_interval=self.logging_interval,
            batch_size=self.batch_size_learn_planet)

        # save the models if desired
        if self.save_model_after_lean_planet:
            self.s4_model.save_noise_model("noise_model_fine_tuned.pkl")
            self.s4_model.save_normalization_model("normalization_model.pkl")
            self.s4_model.save_planet_model("planet_model.pkl")

    def _create_s4_residuals(
            self,
            additional_name: str = "",
            account_for_planet_model: bool = False):

        mean_residual = self.s4_model.compute_residual(
            account_for_planet_model=account_for_planet_model,
            combine="mean")

        median_residual = self.s4_model.compute_residual(
            account_for_planet_model=account_for_planet_model,
            combine="median")

        # 4.) Store everything in the result dict and return it
        result_dict = dict()
        if self.special_name is None:
            result_dict["s4_mean" + additional_name] = mean_residual
            result_dict["s4_median" + additional_name] = median_residual
        else:
            result_dict["s4_mean_" + self.special_name + additional_name] \
                = mean_residual
            result_dict["s4_median_" + self.special_name + additional_name] \
                = median_residual

        return result_dict

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:

        # 1.) Create the S4 model
        if not self.lean_noise_model:
            if self.normalization_model_file is None \
                    or self.noise_model_file is None:
                raise ValueError(
                    "Normalization or noise model file is None. "
                    "You can either learn a noise model from scratch and setup "
                    "all necessary parameters using setup_learning_noise_model "
                    "or give a noise and normalization model in order "
                    "to restore them.")

            # 1.1) If the noise model is given, load it
            self.s4_model = S4.create_from_checkpoint(
                noise_model_file=self.noise_model_file,
                normalization_model_file=self.normalization_model_file,
                s4_work_dir=self.work_dir,
                science_cube=stack_with_fake_planet,
                adi_angles=parang_rad,
                psf_template=psf_template,
                device=self.device,
                verbose=self.verbose,
                planet_convolve_second=self.planet_convolve_second,
                planet_use_up_sample=self.planet_use_up_sample)
        else:
            # 1.2) If no model is given we have to learn a model from scratch
            self._build_s4_noise_model(
                stack_with_fake_planet=stack_with_fake_planet,
                parang_rad=parang_rad,
                psf_template=psf_template)

        # 2.) if fine-tuning with a planet model is desired, do it
        result_dict = dict()
        if self.fine_tune_planet:
            # create the raw residuals if desired
            if self.create_raw_residuals:
                residuals_dict = self._create_s4_residuals(
                    additional_name="_raw",
                    account_for_planet_model=False)
                result_dict.update(residuals_dict)

            self._learn_planet_model()

        # 3.) compute the residual
        if self.fine_tune_planet:
            account_for_planet_model = True
        else:
            account_for_planet_model = False

        residuals_dict = self._create_s4_residuals(
            additional_name="",
            account_for_planet_model=account_for_planet_model)
        result_dict.update(residuals_dict)

        return result_dict
