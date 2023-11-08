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
            device: Union[int, str] = "cpu",
            work_dir: str = None,
            verbose: bool = False):

        self.special_name = special_name
        self.noise_model_file = noise_model_file
        self.device = device
        self.work_dir = work_dir
        self.verbose = verbose

        if self.noise_model_file is not None:
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
        self.train_num_epochs = None
        self.training_learning_rate = None
        self.lean_noise_model = "No"

        # 2.) will be set in setup_leaning_planet_model
        self.rotation_grid_down_sample = None
        self.save_model_after_fit = None

        # will be created once the data is available
        self.s4_model = None

    def setup_create_noise_model_closed_form(
            self,
            lambda_reg: float,
            rotation_grid_down_sample=1,
            noise_cut_radius_psf=None,
            noise_mask_radius=None,
            convolve=True,
            noise_normalization="normal",
            save_models: bool = True,
            train_num_epochs: int = 0,
            learning_rate_fine_tune_noise: float = 1e-6):

        self.noise_cut_radius_psf = noise_cut_radius_psf
        self.noise_mask_radius = noise_mask_radius
        self.convolve = convolve
        self.noise_normalization = noise_normalization
        self.lambda_reg = lambda_reg
        self.train_num_epochs = train_num_epochs
        self.training_learning_rate = learning_rate_fine_tune_noise
        self.save_model_after_fit = save_models
        self.rotation_grid_down_sample = rotation_grid_down_sample
        self.lean_noise_model = "Closed form"

    def setup_create_noise_model_lbfgs(
            self,
            lambda_reg: float,
            rotation_grid_down_sample=1,
            noise_cut_radius_psf=None,
            noise_mask_radius=None,
            convolve=True,
            noise_normalization="normal",
            save_models: bool = True,
            train_num_epochs: int = 0):

        self.noise_cut_radius_psf = noise_cut_radius_psf
        self.noise_mask_radius = noise_mask_radius
        self.convolve = convolve
        self.noise_normalization = noise_normalization
        self.lambda_reg = lambda_reg
        self.train_num_epochs = train_num_epochs
        self.save_model_after_fit = save_models
        self.rotation_grid_down_sample = rotation_grid_down_sample
        self.lean_noise_model = "LBFGS"

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
            psf_template: np.ndarray,
            exp_id):

        # 1.) Create the S4 model
        self.s4_model = S4(
            science_cube=stack_with_fake_planet,
            adi_angles=parang_rad,
            psf_template=psf_template,
            device=self.device,
            work_dir=self.work_dir,
            verbose=self.verbose,
            rotation_grid_subsample=self.rotation_grid_down_sample,
            noise_cut_radius_psf=self.noise_cut_radius_psf,
            noise_mask_radius=self.noise_mask_radius,
            noise_normalization=self.noise_normalization,
            noise_model_lambda_init=self.lambda_reg,
            noise_model_convolve=self.convolve)

        if self.lean_noise_model == "LBFGS":
            self.s4_model.fit_noise_model(
                num_epochs=self.train_num_epochs,
                use_rotation_loss=True,
                training_name=exp_id,
                logging_interval=5)

        elif self.lean_noise_model == "Closed form":
            self.s4_model.fit_noise_model_closed_form(
                num_epochs=self.train_num_epochs,
                training_name=exp_id,
                logging_interval=5,
                learning_rate=self.training_learning_rate)

    def _create_s4_residuals(self):

        mean_residual = self.s4_model.compute_residual(
            combine="mean")

        median_residual = self.s4_model.compute_residual(
            combine="median")

        # 4.) Store everything in the result dict and return it
        result_dict = dict()
        if self.special_name is None:
            result_dict["s4_mean"] = mean_residual
            result_dict["s4_median"] = median_residual
        else:
            result_dict["s4_mean_" + self.special_name] = mean_residual
            result_dict["s4_median_" + self.special_name] = median_residual

        return result_dict

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:

        # 1.) Create the S4 model
        if self.lean_noise_model == "No":
            if self.noise_model_file is None:
                raise ValueError(
                    "Noise model file is None. "
                    "You can either learn a noise model from scratch and setup "
                    "all necessary parameters using "
                    "setup_create_noise_model_lbfgs "
                    "or give a noise model in order to restore it.")

            # 1.1) If the noise model is given, load it
            self.s4_model = S4.create_from_checkpoint(
                noise_model_file=self.noise_model_file,
                normalization_model_file=None,
                s4_work_dir=self.work_dir,
                science_cube=stack_with_fake_planet,
                adi_angles=parang_rad,
                psf_template=psf_template,
                device=self.device,
                verbose=self.verbose)
        else:
            # 1.2) If no model is given we have to learn a model from scratch
            self._build_s4_noise_model(
                stack_with_fake_planet=stack_with_fake_planet,
                parang_rad=parang_rad,
                psf_template=psf_template,
                exp_id=exp_id)

        # 2.) compute the residual
        return self._create_s4_residuals()
