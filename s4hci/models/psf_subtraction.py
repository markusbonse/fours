from typing import Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from s4hci.models.noise import S4Noise
from s4hci.models.normalization import S4FrameNormalization
from s4hci.utils.adi_tools import combine_residual_stack
from s4hci.utils.data_handling import save_as_fits


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

        # 3.) Create normalization model
        self.normalization_model = S4FrameNormalization(
            image_size=self.data_image_size,
            normalization_type=noise_normalization)
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
    def find_closed_form_noise_model(
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

    @staticmethod
    def _normalize_for_tensorboard(frame_in):
        image_for_tb = deepcopy(frame_in)
        image_for_tb -= np.min(image_for_tb)
        image_for_tb /= np.max(image_for_tb)
        return image_for_tb

    def _create_tensorboard_logger(self):
        time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        self.fine_tune_start_time = time_str
        current_logdir = self.tensorboard_dir / \
            Path(self.fine_tune_start_time)
        current_logdir.mkdir()
        self.tensorboard_logger = SummaryWriter(current_logdir)

    @_print_progress("S4 model: fine tuning noise model")
    def fine_tune_noise_model(
            self,
            num_epochs,
            learning_rate=1e-6,
            batch_size=-1):

        if self.work_dir is not None:
            self._create_tensorboard_logger()

        # 1.) normalize the science data
        x_norm = self.normalization_model(self.science_cube)
        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)

        # 2.) move models to the GPU
        self.noise_model = self.noise_model.to(self.device)

        # 3.) Create the optimizer and add the parameters we want to optimize
        optimizer = optim.Adam(
            [self.noise_model.betas_raw, ],
            lr=learning_rate)

        # 4.) Create the DataLoader
        if batch_size == -1:
            batch_size = x_norm.shape[0]
            # upload the data to the device
            science_norm_flatten = science_norm_flatten.to(self.device)

        data_loader = DataLoader(
            science_norm_flatten,
            batch_size=batch_size,
            shuffle=True)

        # 5.) Run the fine-tuning
        # needed for gradient accumulation in order to normalize the loss
        num_steps_per_epoch = len(data_loader)

        epoch_range = tqdm(range(num_epochs)) if \
            self.noise_model.verbose else range(num_epochs)

        for epoch in epoch_range:
            optimizer.zero_grad()

            # we have to keep track of the loss sum within gradient accumulation
            running_reg_loss = 0
            running_recon_loss = 0

            for tmp_frames in data_loader:
                # 0.) upload tmp_frames if needed
                if tmp_frames.device != torch.device(self.device):
                    tmp_frames = tmp_frames.to(self.device)

                # 1.) run the forward path of the noise model
                self.noise_model.compute_betas()
                noise_estimate = self.noise_model(tmp_frames)

                # 2.) Compute the loss
                loss_recon = ((noise_estimate - tmp_frames) ** 2).sum()
                loss_reg = (self.noise_model.betas_raw ** 2).sum() \
                    * self.noise_model.lambda_reg \
                    / num_steps_per_epoch

                # 3.) Backward
                loss = loss_recon + loss_reg
                loss.backward()

                # 4.) Track the current loss
                running_reg_loss += loss_reg.detach().item()
                running_recon_loss += loss_recon.detach().item()

            # Make one accumulated gradient step
            optimizer.step()

            # 5.) Logg the information
            self._logg_loss_values(
                epoch=epoch,
                loss_recon=running_recon_loss,
                loss_reg=running_reg_loss)

        # 7.) Clean up GPU
        self.noise_model = self.noise_model.cpu()
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
        residual_after_fine_tuning = combine_residual_stack(
            residual_stack=residual_stack,
            angles=self.adi_angles,
            combine=combine,
            subtract_temporal_average=False,
            num_cpus=num_cpus)

        return residual_after_fine_tuning
