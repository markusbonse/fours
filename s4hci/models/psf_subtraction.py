from typing import Union
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from s4hci.models.noise import S4Noise
from s4hci.models.normalization import S4FrameNormalization
from s4hci.models.rotation import FieldRotationModel
from s4hci.utils.adi_tools import combine_residual_stack
from s4hci.utils.data_handling import save_as_fits
from s4hci.utils.logging import normalize_for_tensorboard


class S4:

    def __init__(
            self,
            science_cube,
            adi_angles,
            psf_template,
            noise_cut_radius_psf,
            noise_mask_radius,
            device=0,
            work_dir=None,
            noise_normalization="normal",
            noise_model_lambda_init=1e3,
            noise_model_convolve=True,
            rotation_grid_subsample=1,
            verbose=True
    ):
        # 1.) Save all member data
        self.device = device
        self.adi_angles = adi_angles
        self.science_cube = torch.from_numpy(science_cube).float()
        self.psf_template = psf_template
        self.data_image_size = self.science_cube.shape[-1]

        if work_dir is not None:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = None
        self.residuals_dir, self.tensorboard_dir, self.models_dir = \
            self._setup_working_dir()

        # 2.) Create the noise model
        self.noise_model = S4Noise(
            data_image_size=self.data_image_size,
            psf_template=self.psf_template,
            lambda_reg=noise_model_lambda_init,
            cut_radius_psf=noise_cut_radius_psf,
            # TODO the noise mask radius options should be removed
            mask_template_setup=("radius", noise_mask_radius),
            convolve=noise_model_convolve,
            verbose=verbose).float()

        # 2.1) Create the rotation model
        self.rotation_model = FieldRotationModel(
            all_angles=self.adi_angles,
            input_size=self.data_image_size,
            subsample=rotation_grid_subsample,
            inverse=False,
            register_grid=True).float()

        # 3.) Create normalization model
        self.normalization_model = S4FrameNormalization(
            image_size=self.data_image_size,
            normalization_type=noise_normalization).float()
        self.normalization_model.prepare_normalization(
            science_data=self.science_cube)

        # 5.) Create the tensorboard logger for the fine_tuning
        self.tensorboard_logger = None
        self.fine_tune_start_time = None

    @classmethod
    def create_from_checkpoint(
            cls,
            noise_model_file: str,
            normalization_model_file: str,
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
                if self.noise_model.verbose:
                    print(msg + " ... ", end='')
                    result = function(self, *args, **kwargs)
                    print("[DONE]")
                else:
                    result = function(self, *args, **kwargs)
                return result
            return wrapper
        return decorator

    def _setup_working_dir(self):
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

    @staticmethod
    def _check_model_dir(function):
        def check_workdir(self, *args, **kwargs):
            if self.models_dir is None:
                raise FileNotFoundError(
                    "Saving the model requires a work directory.")
            function(self, *args, **kwargs)
        return check_workdir

    @_check_model_dir
    @_print_progress("S4 model: saving model")
    def save_models(
            self,
            file_name_noise_model,
            file_name_normalization_model):
        self.normalization_model.save(
            self.models_dir / file_name_normalization_model)
        self.noise_model.save(
            self.models_dir / file_name_noise_model)

    @_print_progress("S4 model: restoring models")
    def restore_models(
            self,
            file_noise_model=None,
            file_normalization_model=None,
            verbose=False):

        if file_noise_model is not None:
            self.noise_model = S4Noise.load(
                file_noise_model,
                verbose)

        if file_normalization_model is not None:
            self.normalization_model = S4FrameNormalization.load(
                file_normalization_model)

    @_print_progress("S4 model: validating noise model")
    def validate_lambdas_noise(
            self,
            num_separations,
            lambdas,
            num_test_positions,
            test_size=0.3,
            approx_svd=5000):
        """
        First processing step
        """

        # 1.) split the data into training and test data
        if isinstance(test_size, float):
            train_idx, test_idx, = train_test_split(
                np.arange(len(self.science_cube)),
                test_size=test_size,
                random_state=42,
                shuffle=True)

            x_train = self.science_cube[train_idx]
            x_test = self.science_cube[test_idx]
        else:
            # Use an even/odd split
            x_train = self.science_cube[0::2]
            x_test = self.science_cube[1::2]

        # 2.) Normalize the training and test data
        tmp_normalization = S4FrameNormalization(
            image_size=self.data_image_size,
            normalization_type=self.normalization_model.normalization_type)
        tmp_normalization.prepare_normalization(x_train)
        x_test = tmp_normalization(x_test)
        x_train = tmp_normalization(x_train)

        # 3.) validate the lambda values of the noise model
        all_results, best_lambda = self.noise_model.validate_lambdas(
            num_separations=num_separations,
            lambdas=lambdas,
            science_data_train=x_train,
            science_data_test=x_test,
            num_test_positions=num_test_positions,
            approx_svd=approx_svd,
            device=self.device)

        return all_results, best_lambda

    @_print_progress("S4 model: finding closed form noise model")
    def _find_closed_form_noise_model(
            self,
            fp_precision="float32"):
        """
        Second processing step
        """

        # 1.) normalize the data
        x_train = self.normalization_model(self.science_cube)

        # 2.) Train the noise model
        self.noise_model.fit(
            x_train,
            device=self.device,
            fp_precision=fp_precision)

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
            residual_median):

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
            Path(self.fine_tune_start_time)
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

    @_print_progress("S4 model: Find noise model closed form and fine tune.")
    def fit_noise_model_closed_form(
            self,
            num_epochs,
            training_name="",
            logging_interval=1,
            learning_rate=1e-6):

        # 1.) find the closed form solution
        self._find_closed_form_noise_model()

        # 2.) fix numerical issues with Gradient Descent
        optimizer = optim.Adam
        optimizer_kwargs = {"lr": learning_rate}
        self.fit_noise_model(
            num_epochs=num_epochs,
            use_rotation_loss=False,
            logging_interval=logging_interval,
            training_name=training_name,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs)

    @_print_progress("S4 model: Fit noise model")
    def fit_noise_model(
            self,
            num_epochs,
            use_rotation_loss,
            training_name="",
            logging_interval=1,
            optimizer=None,
            optimizer_kwargs=None):

        # Create the tensorboard logger
        if self.work_dir is not None:
            self._create_tensorboard_logger(training_name)

        # 1.) normalize the science data
        x_norm = self.normalization_model(self.science_cube)
        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)

        # 2.) move models to the GPU
        self.noise_model = self.noise_model.to(self.device)
        self.rotation_model = self.rotation_model.to(self.device)
        science_norm_flatten = science_norm_flatten.to(self.device)

        # 3.) Create the optimizer and add the parameters we want to optimize
        if optimizer is not None:
            optimizer = optimizer(
                [self.noise_model.betas_raw, ],
                **optimizer_kwargs)
        else:
            if optimizer_kwargs is None:
                optimizer_kwargs = {
                    "max_iter": 20,
                    "history_size": 10}

            optimizer = optim.LBFGS(
                [self.noise_model.betas_raw, ],
                **optimizer_kwargs)

        # 5.) Run the optimization
        for epoch in tqdm(range(num_epochs)):

            def full_closure(compute_residuals):
                optimizer.zero_grad()

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

                rotated_frames = self.rotation_model(
                    residual_sequence,
                    parang_idx=torch.arange(len(residual_sequence)))

                # 2.) Compute the loss
                if use_rotation_loss:
                    loss_recon = torch.var(rotated_frames, axis=0).sum()
                    loss_recon *= rotated_frames.shape[0]
                else:
                    loss_recon = ((science_norm_flatten -
                                   noise_estimate)**2).sum()

                loss_reg = (self.noise_model.betas_raw ** 2).sum() \
                    * self.noise_model.lambda_reg

                # 3.) Backward
                loss = loss_recon + loss_reg
                loss.backward()

                if compute_residuals:
                    residual = torch.mean(rotated_frames, axis=0)[
                        0].detach().cpu().numpy()
                    residual_median = torch.median(rotated_frames, axis=0)[0][
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
                    residual_median=current_residual_median)

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
        torch.cuda.empty_cache()

    @_print_progress("S4 model: computing residual")
    @torch.no_grad()
    def compute_residual(
            self,
            combine="median",
            num_cpus=8
    ):
        # 1.) normalize science data
        x_norm = self.normalization_model(self.science_cube)
        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)

        # 2.) compute the noise estimate
        noise_estimate = self.noise_model(science_norm_flatten)

        # 3.) compute the residual sequence
        residual_sequence = science_norm_flatten - noise_estimate
        residual_stack = residual_sequence.view(
            self.science_cube.shape[0],
            self.noise_model.image_size,
            self.noise_model.image_size).detach().cpu().numpy()

        # 4.) Compute the residual image
        residual_image = combine_residual_stack(
            residual_stack=residual_stack,
            angles=self.adi_angles,
            combine=combine,
            subtract_temporal_average=False,
            num_cpus=num_cpus)

        return residual_image
