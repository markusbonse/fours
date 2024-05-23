from typing import Union
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from fours.models.noise import FourSNoise
from fours.models.normalization import FourSFrameNormalization
from fours.models.rotation import FieldRotationModel

from fours.utils.data_handling import save_as_fits
from fours.utils.logging import normalize_for_tensorboard
from fours.utils.fwhm import get_fwhm
from fours.utils.adi_tools import combine_residual_stack


class FourS:

    def __init__(
            self,
            science_cube,
            adi_angles,
            psf_template,
            noise_model_lambda,
            psf_fwhm=None,
            right_reason_mask_factor=1.5,
            rotation_grid_subsample=1,
            device=0,
            work_dir=None,
            verbose=True,
    ):
        # 0.) If some parameters are not given, set them to default values
        if psf_fwhm is None:
            psf_fwhm = get_fwhm(psf_template)
        right_reason_mask_radius = psf_fwhm * right_reason_mask_factor

        # 1.) Save all member data
        self.right_reason_mask_factor = right_reason_mask_factor
        self.device = device
        self.verbose = verbose
        self.adi_angles = adi_angles
        self.science_cube = torch.from_numpy(science_cube).float()
        self.psf_template = psf_template
        self.data_image_size = self.science_cube.shape[-1]

        if work_dir is not None:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = None
        self.residuals_dir, self.tensorboard_dir, self.models_dir = \
            self._setup_work_dir()

        # 2.) Create the noise model
        self.noise_model = FourSNoise(
            data_image_size=self.data_image_size,
            psf_template=self.psf_template,
            lambda_reg=noise_model_lambda,
            cut_radius_psf=psf_fwhm,
            right_reason_mask_radius=right_reason_mask_radius,
            convolve=True).float()

        # 2.1) Create the rotation model
        self.rotation_model = FieldRotationModel(
            all_angles=self.adi_angles,
            input_size=self.data_image_size,
            subsample=rotation_grid_subsample,
            inverse=False,
            register_grid=True).float()

        # 3.) Create normalization model
        self.normalization_model = FourSFrameNormalization(
            image_size=self.data_image_size,
            normalization_type="normal").float()
        self.normalization_model.prepare_normalization(
            science_data=self.science_cube)

        # 5.) Create the tensorboard logger for the fine_tuning
        self.tensorboard_logger = None
        self.fine_tune_start_time = None

    @classmethod
    def create_from_checkpoint(
            cls,
            noise_model_file: str,
            normalization_model_file: Union[str, Path, None],
            s4_work_dir: Union[str, Path, None],
            science_cube: np.ndarray,
            adi_angles: np.ndarray,
            psf_template: np.ndarray,
            device: Union[int, str],
            verbose: bool = True):

        # create the s4 model
        s4_model = cls(
            science_cube=science_cube,
            adi_angles=adi_angles,
            psf_template=psf_template,
            noise_cut_radius_psf=1,  # will be restored
            noise_mask_radius=1,  # will be restored
            device=device,
            negative_wing_suppression=False,  # will be restored
            noise_model_convolve=True,  # will be restored
            noise_normalization="normal",  # will be restored
            work_dir=s4_work_dir,
            verbose=verbose)

        # restore the noise and normalization model
        s4_model.restore_models(
            file_noise_model=noise_model_file,
            file_normalization_model=normalization_model_file,
            verbose=verbose)

        return s4_model

    @staticmethod
    def _print_progress(msg):
        def decorator(function):
            def wrapper(self, *args, **kwargs):
                if self.verbose:
                    print(msg + " ... ", end='')
                    result = function(self, *args, **kwargs)
                    print("[DONE]")
                else:
                    result = function(self, *args, **kwargs)
                return result
            return wrapper
        return decorator

    def _setup_work_dir(self):
        if self.work_dir is None:
            return None, None, None

        # make sure the working dir is a dir
        self.work_dir.mkdir(exist_ok=True)

        residuals_dir = self.work_dir / "residuals"
        tensorboard_dir = self.work_dir / "tensorboard"
        models_dir = self.work_dir / "models"

        residuals_dir.mkdir(exist_ok=True)
        tensorboard_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        return residuals_dir, tensorboard_dir, models_dir

    @_print_progress("4S model: saving model")
    def save_models(
            self,
            file_name_noise_model,
            file_name_normalization_model):

        if self.models_dir is None:
            raise FileNotFoundError(
                "Saving the model requires a work directory.")

        self.normalization_model.save(
            self.models_dir / file_name_normalization_model)
        self.noise_model.save(
            self.models_dir / file_name_noise_model)

    @_print_progress("4S model: restoring models")
    def restore_models(
            self,
            file_noise_model=None,
            file_normalization_model=None):

        if file_noise_model is not None:
            self.noise_model = FourSNoise.load(file_noise_model)

        if file_normalization_model is not None:
            self.normalization_model = FourSFrameNormalization.load(
                file_normalization_model)

    def _logg_loss_values(
            self,
            epoch,
            loss_recon,
            loss_reg):

        if self.work_dir is None:
            return

        self.tensorboard_logger.add_scalar(
            "Loss/Reconstruction_loss",
            loss_recon,
            epoch)

        self.tensorboard_logger.add_scalar(
            "Loss/Regularization_loss",
            loss_reg,
            epoch)

    def _logg_residuals(
            self,
            epoch,
            residual_mean,
            residual_median,
            training_name):

        self.tensorboard_logger.add_image(
            "Images/Residual",
            normalize_for_tensorboard(residual_mean),
            epoch,
            dataformats="HW")

        self.tensorboard_logger.add_image(
            "Images/Residual_Median",
            normalize_for_tensorboard(residual_median),
            epoch,
            dataformats="HW")

        tmp_residual_dir = self.residuals_dir / \
            Path(training_name + "_" + self.fine_tune_start_time)
        tmp_residual_dir.mkdir(exist_ok=True)

        save_as_fits(
            residual_mean,
            tmp_residual_dir /
            Path("Residual_Mean_epoch_" + str(epoch).zfill(4)
                 + ".fits"),
            overwrite=True)

        save_as_fits(
            residual_median,
            tmp_residual_dir /
            Path("Residual_Median_epoch_" + str(epoch).zfill(4)
                 + ".fits"),
            overwrite=True)

    def _create_tensorboard_logger(self, training_name):
        time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        self.fine_tune_start_time = time_str
        current_logdir = self.tensorboard_dir / \
            Path(training_name + "_" + self.fine_tune_start_time)
        current_logdir.mkdir()
        self.tensorboard_logger = SummaryWriter(current_logdir)

    def _get_residual_sequence(self):
        # 0.) Normalize the science data
        x_norm = self.normalization_model(self.science_cube)
        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)

        # 1.) run the forward path of the noise model
        self.noise_model.compute_betas()
        noise_estimate = self.noise_model(science_norm_flatten)

        # 2.) compute the residual and rotate it
        residual_sequence = science_norm_flatten - noise_estimate
        residual_sequence = residual_sequence.view(
            residual_sequence.shape[0],
            1,
            self.data_image_size,
            self.data_image_size)

        rotated_residual_sequence = self.rotation_model(
            residual_sequence,
            parang_idx=torch.arange(len(residual_sequence)))

        return rotated_residual_sequence, residual_sequence

    @_print_progress("S4 model: Fit noise model")
    def fit_noise_model(
            self,
            num_epochs,
            training_name="",
            logging_interval=1,
            optimizer=None,
            optimizer_kwargs=None):

        # 1.) Create the tensorboard logger
        if self.work_dir is not None:
            self._create_tensorboard_logger(training_name)

        # 2.) move models to the GPU
        self.noise_model = self.noise_model.to(self.device)
        self.rotation_model = self.rotation_model.to(self.device)
        self.science_cube = self.science_cube.to(self.device)
        self.normalization_model = self.normalization_model.to(self.device)

        # 3.) Create the optimizer and add the parameters we want to optimize
        trainable_params = [self.noise_model.betas_raw, ]

        if optimizer is not None:
            optimizer = optimizer(
                trainable_params,
                **optimizer_kwargs)
        else:
            if optimizer_kwargs is None:
                optimizer_kwargs = {
                    "max_iter": 20,
                    "history_size": 10}

            optimizer = optim.LBFGS(
                trainable_params,
                **optimizer_kwargs)

        # 4.) Run the optimization
        for epoch in tqdm(range(num_epochs)):

            def full_closure(compute_residuals):
                optimizer.zero_grad()

                # 4.1) Get the residual sequence
                rotated_residual_sequence, residual_sequence = (
                    self._get_residual_sequence())

                # 4.2) Compute the loss
                loss_recon = torch.var(rotated_residual_sequence, axis=0).sum()
                loss_recon *= rotated_residual_sequence.shape[0]

                loss_reg = (self.noise_model.betas_raw ** 2).sum() \
                    * self.noise_model.lambda_reg

                # 4.3) Backward
                loss = loss_recon + loss_reg
                loss.backward()

                if compute_residuals:
                    residual = torch.mean(
                        rotated_residual_sequence, axis=0)[
                        0].detach().cpu().numpy()
                    residual_median = torch.median(
                        rotated_residual_sequence, axis=0)[0][
                        0].detach().cpu().numpy()

                    return loss_recon, loss_reg, residual, residual_median
                else:
                    return loss, loss_recon, loss_reg

            def loss_closure():
                return full_closure(False)[0]

            optimizer.step(loss_closure)

            # 5.) Logg the information
            if epoch % logging_interval == 0:
                # Logg the full information with residuals
                # x.) get current residuals
                current_loss_recon, \
                    current_loss_reg, \
                    current_residual, \
                    current_residual_median = full_closure(True)

                self._logg_residuals(
                    epoch=epoch,
                    residual_mean=current_residual,
                    residual_median=current_residual_median,
                    training_name=training_name)

            else:
                # Logg the loss information
                _, current_loss_recon, current_loss_reg = full_closure(False)

            self._logg_loss_values(
                epoch=epoch,
                loss_recon=current_loss_recon,
                loss_reg=current_loss_reg)

        # 7.) Clean up GPU
        self.noise_model = self.noise_model.cpu()
        self.rotation_model = self.rotation_model.cpu()
        self.normalization_model = self.normalization_model.cpu()
        self.science_cube = self.science_cube.cpu()
        torch.cuda.empty_cache()

    @_print_progress("S4 model: computing residual")
    @torch.no_grad()
    def compute_residuals(self, num_cpus=4):

        # 1.) Get the residual sequence
        rotated_residual_sequence, residual_sequence = (
            self._get_residual_sequence())

        # 2.) Compute the residual image (mean)
        mean_residual = torch.mean(rotated_residual_sequence,
                                   axis=0)[0].cpu().numpy()

        # 3.) Compute the residual image (median)
        median_residual = torch.median(rotated_residual_sequence,
                                       axis=0)[0][0].cpu().numpy()

        return mean_residual, median_residual
