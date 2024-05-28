from typing import List, Dict, Union
import numpy as np
from pathlib import Path
from datetime import datetime
from applefy.detections.contrast import DataReductionInterface

from fours.models.psf_subtraction import FourS
from fours.utils.pca import pca_psf_subtraction_gpu, pca_tensorboard_logging
from fours.utils.adi_tools import cadi_psf_subtraction, cadi_psf_subtraction_gpu


class CADIDataReduction(DataReductionInterface):

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


class CADIDataReductionGPU(DataReductionInterface):

    def __init__(self, device):
        self.device = device

    def get_method_keys(self) -> List[str]:
        return ["cADI", ]

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str):

        residual = cadi_psf_subtraction_gpu(
            images=stack_with_fake_planet,
            angles=parang_rad,
            device=self.device)

        result_dict = dict()
        result_dict["cADI"] = residual

        return result_dict


class PCADataReductionGPU(DataReductionInterface):

    def __init__(
            self,
            pca_numbers: np.ndarray,
            approx_svd: int,
            work_dir: Union[str, Path] = None,
            special_name: str = None,
            device: Union[int, str] = "cpu",
            verbose: bool = False):
        self.pca_numbers = pca_numbers
        self.approx_svd = approx_svd
        self.device = device
        self.verbose = verbose
        if work_dir is not None:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = None

        if special_name is None:
            self.special_name = ""
        else:
            self.special_name = special_name

    def get_method_keys(self) -> List[str]:

        keys = [self.special_name + "_PCA_" + str(num_pcas).zfill(3) +
                "_components" for num_pcas in self.pca_numbers]

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

        if self.work_dir is not None:
            time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            current_logdir = self.work_dir / \
                Path(exp_id + "_" + self.special_name + "_PCA_" + time_str)
            current_logdir.mkdir(exist_ok=True, parents=True)

            pca_tensorboard_logging(
                log_dir=current_logdir,
                pca_residuals=pca_residuals,
                pca_numbers=self.pca_numbers)

        result_dict = dict()
        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = pca_residuals[idx]

        return result_dict


class FourSDataReduction(DataReductionInterface):
    def __init__(
            self,
            device,
            lambda_reg: float,
            psf_fwhm=None,
            right_reason_mask_factor: float = 1.5,
            rotation_grid_down_sample=1,
            logging_interval: int = 1,
            save_models: bool = True,
            train_num_epochs: int = 500,
            special_name: str = None,
            work_dir: str = None,
            verbose: bool = False):

        # 0.) parameters for the wrapper
        self.special_name = special_name
        self.device = device
        self.work_dir = work_dir
        self.verbose = verbose

        # 1.) parameters for 4S
        self.psf_fwhm = psf_fwhm
        self.right_reason_mask_factor = right_reason_mask_factor
        self.lambda_reg = lambda_reg
        self.rotation_grid_down_sample = rotation_grid_down_sample
        self.save_model_after_fit = save_models
        self.train_num_epochs = train_num_epochs
        self.logging_interval = logging_interval

        # will be created once the data is available
        self.fours_model = None

    def get_method_keys(self) -> List[str]:
        if self.special_name is None:
            return ["s4_mean", "s4_median"]
        else:
            return ["s4_mean_" + self.special_name,
                    "s4_median_" + self.special_name]

    def _build_4s_noise_model(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id):

        # 1.) Create the 4S model
        self.fours_model = FourS(
            science_cube=stack_with_fake_planet,
            adi_angles=parang_rad,
            psf_template=psf_template,
            noise_model_lambda=self.lambda_reg,
            psf_fwhm=self.psf_fwhm,
            right_reason_mask_factor=self.right_reason_mask_factor,
            rotation_grid_subsample=self.rotation_grid_down_sample,
            device=self.device,
            work_dir=self.work_dir,
            verbose=self.verbose)

        name = exp_id
        if self.special_name is not None:
            name += "_" + self.special_name
        self.fours_model.fit_noise_model(
            num_epochs=self.train_num_epochs,
            training_name=name,
            logging_interval=self.logging_interval)

    def _create_4s_residuals(self):

        mean_residual, median_residual = self.fours_model.compute_residuals()

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
        self._build_4s_noise_model(
            stack_with_fake_planet=stack_with_fake_planet,
            parang_rad=parang_rad,
            psf_template=psf_template,
            exp_id=exp_id)

        if self.save_model_after_fit:
            name = ""
            if self.special_name is not None:
                name = "_" + self.special_name
            self.fours_model.save_models(
                file_name_noise_model=
                "noise_model_" + exp_id + name + ".pkl",
                file_name_normalization_model=
                "normalization_model_" + exp_id + name + ".pkl")

        # 2.) compute the residual
        return self._create_4s_residuals()
