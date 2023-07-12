from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from s4hci.models.noise import S4Noise
from s4hci.models.planet import S4Planet
from s4hci.utils.adi_tools import combine_residual_stack
from s4hci.utils.data_handling import save_as_fits


class S4:

    def __init__(
            self,
            data_cube,
            parang,
            psf_template,
            noise_noise_cut_radius_psf,
            noise_mask_radius,
            device=0,
            noise_lambda_init=1e3,
            planet_convolve_second=True,
            planet_use_up_sample=1,
            work_dir=None,
            verbose=True
    ):
        # 1.) Save all member data
        self.device = device
        self.parang = parang
        self.data_cube = data_cube
        self.psf_template = psf_template
        self.data_image_size = data_cube.shape[-1]
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
            lambda_reg=noise_lambda_init,
            cut_radius_psf=noise_noise_cut_radius_psf,
            mask_template_setup=("radius", noise_mask_radius),
            convolve=True,
            verbose=verbose).float()

        # 3.) Create the planet model
        self.planet_model = S4Planet(
            data_image_size=self.data_image_size,
            psf_template=self.psf_template,
            convolve_second=planet_convolve_second,
            inner_mask_radius=0,
            use_up_sample=planet_use_up_sample).float()

        # 4.) Create the tensorboard logger for the fine_tuning
        self.tensorboard_logger = None
        self.fine_tune_start_time = None

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
        x_train, x_test, = train_test_split(
            self.data_cube,
            test_size=test_size,
            random_state=42,
            shuffle=False)

        # 2.) validate the lambda values of the noise model
        x_train = torch.from_numpy(x_train).float().to(self.device)
        x_test = torch.from_numpy(x_test).float().to(self.device)
        self.noise_model = self.noise_model.to(self.device)

        all_results, best_lambda = self.noise_model.validate_lambdas(
            num_separations=num_separations,
            lambdas=lambdas,
            science_data_train=x_train,
            science_data_test=x_test,
            num_test_positions=num_test_positions,
            approx_svd=approx_svd)

        # 3.) free gpu space
        del x_train, x_test
        self.noise_model = self.noise_model.cpu()

        # 4.) save the model
        self.noise_model.save(self.models_dir / "noise_model_lambda_val.pkl")

        return all_results, best_lambda

    def find_closed_form_noise_model(self):
        """
        Second processing step
        """

        # 1.) Train the noise model
        x_train = torch.from_numpy(self.data_cube).float().to(self.device)
        self.noise_model = self.noise_model.to(self.device)
        self.noise_model.fit(x_train)

        # 2.) clean up
        del x_train
        self.noise_model = self.noise_model.cpu()

        # 2.) Save the noise model
        self.noise_model.save(self.models_dir / "noise_model_closed_form.pkl")

    def _logg_fine_tune_status(
            self,
            epoch,
            loss_reg,
            start_reg_loss,
            loss_recon,
            start_recon_loss,
            logging_interval,
            planet_signal):

        if self.work_dir is None:
            return

        self.tensorboard_logger.add_scalar(
            "Loss/Reconstruction_delta",
            loss_recon.item() - start_recon_loss,
            epoch)

        self.tensorboard_logger.add_scalar(
            "Loss/Regularization_delta",
            loss_reg.item() - start_reg_loss,
            epoch)

        if not epoch % logging_interval == logging_interval - 1:
            return

        with torch.no_grad():
            tmp_frame = planet_signal.detach()[-1, 0].cpu().numpy()
            self.tensorboard_logger.add_image(
                "Images/Planet_signal_estimate",
                self.normalize_for_tensorboard(tmp_frame),
                epoch,
                dataformats="HW")

            tmp_residual_dir = self.residuals_dir / \
                Path(self.fine_tune_start_time)
            tmp_residual_dir.mkdir(exist_ok=True)

            save_as_fits(
                tmp_frame,
                tmp_residual_dir /
                Path("Planet_signal_estimate_epoch_" + str(epoch).zfill(4)
                     + ".fits"),
                overwrite=True)

            tmp_frame = self.planet_model.get_planet_signal()
            tmp_frame = tmp_frame.detach()[0].cpu().numpy()
            self.tensorboard_logger.add_image(
                "Images/Planet_raw_parameters",
                self.normalize_for_tensorboard(tmp_frame),
                epoch,
                dataformats="HW")

            save_as_fits(
                tmp_frame,
                tmp_residual_dir /
                Path("Planet_raw_parameters_" + str(epoch).zfill(4)
                     + ".fits"),
                overwrite=True)

            self.noise_model.compute_betas()
            betas = self.noise_model.prev_betas.detach().cpu().numpy()
            beta_frame = np.abs(betas[6500].reshape(
                self.data_image_size, self.data_image_size))

            self.tensorboard_logger.add_image(
                "Images/Noise_model_reasons",
                self.normalize_for_tensorboard(beta_frame),
                epoch,
                dataformats="HW")

    def fine_tune_model_with_planet(
            self,
            num_epochs,
            learning_rate_planet=1e-3,
            learning_rate_noise=1e-6,
            fine_tune_noise_model=False,
            rotation_grid_down_sample=10,
            upload_rotation_grid=True,
            logging_interval=10,
    ):
        """
        This is the last step of optimization
        """

        # 1.) setup planet model for training
        self.planet_model.setup_for_training(
            all_angles=self.parang,
            rotation_grid_down_sample=rotation_grid_down_sample,
            upload_rotation_grid=upload_rotation_grid)

        if self.work_dir is not None:
            time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            self.fine_tune_start_time = time_str
            current_logdir = self.tensorboard_dir /\
                Path(self.fine_tune_start_time)
            current_logdir.mkdir()
            self.tensorboard_logger = SummaryWriter(current_logdir)

        # 2.) move models to the GPU
        self.planet_model = self.planet_model.to(self.device)
        self.noise_model = self.noise_model.to(self.device)

        # 3.) set up the normalization
        x_train = torch.from_numpy(self.data_cube).float()
        x_mu = torch.mean(x_train, axis=0)
        x_std = torch.std(x_train, axis=0)
        x_norm = (x_train - x_mu) / x_std

        # move data to GPU
        x_norm = x_norm.to(self.device)
        x_std = x_std.to(self.device)

        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)

        # 4.) Create the optimizer
        if fine_tune_noise_model:
            parameters = [
                {"params": self.noise_model.betas_raw,
                 'lr': learning_rate_noise},
                {"params": self.planet_model.planet_model,
                 'lr': learning_rate_planet}
            ]
        else:
            self.noise_model.betas_raw.requires_grad = False
            parameters = [
                {"params": self.planet_model.planet_model,
                 'lr': learning_rate_planet}
            ]

        # The default learning rate is not needed
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate_planet)

        # if the noise model is not fine-tuned we can compute the noise estimate
        # once at the start of the training loop
        if not fine_tune_noise_model:
            noise_estimate = self.noise_model(science_norm_flatten)
        else:
            noise_estimate = None

        # The index list is needed to get all planet frames during fine-tuning.
        planet_model_idx = torch.from_numpy(np.arange(x_norm.shape[0]))

        # 5.) Run the fine-tuning
        if self.noise_model.verbose:
            epoch_range = tqdm(range(num_epochs))
        else:
            epoch_range = range(num_epochs)

        # values needed to plot the difference in loss
        start_reg_loss = 0
        start_recon_loss = 0

        for epoch in epoch_range:
            optimizer.zero_grad()

            # 1.) Get the current planet signal estimate
            planet_signal = self.planet_model.forward(planet_model_idx)

            # 2.) normalize and reshape the planet signal
            planet_signal = planet_signal / x_std
            planet_signal_norm = planet_signal.view(x_norm.shape[0], -1)

            # 3.) run the forward path
            if fine_tune_noise_model:
                self.noise_model.compute_betas()
                noise_estimate = self.noise_model(science_norm_flatten)

            p_hat_n = self.noise_model(planet_signal_norm)
            p_hat_n[p_hat_n > 0] = 0

            # 4.) Compute the loss
            loss_recon = ((noise_estimate - p_hat_n + planet_signal_norm
                           - science_norm_flatten) ** 2).mean()

            loss_reg = (self.noise_model.betas_raw ** 2).mean() \
                * self.noise_model.lambda_reg

            # 5.) Backward
            loss = loss_recon + loss_reg
            loss.backward()

            optimizer.step()

            # 6.) Logg the information
            if epoch == 0:
                start_reg_loss = loss_reg.item()
                start_recon_loss = loss_recon.item()

            self._logg_fine_tune_status(
                epoch=epoch,
                loss_reg=loss_reg,
                start_reg_loss=start_reg_loss,
                loss_recon=loss_recon,
                start_recon_loss=start_recon_loss,
                logging_interval=logging_interval,
                planet_signal=planet_signal)

        # 8.) Clean up GPU
        del x_norm, x_std
        self.noise_model = self.noise_model.cpu()
        self.planet_model = self.planet_model.cpu()
        torch.cuda.empty_cache()

    @staticmethod
    def normalize_for_tensorboard(frame_in):
        image_for_tb = deepcopy(frame_in)
        image_for_tb -= np.min(image_for_tb)
        image_for_tb /= np.max(image_for_tb)
        return image_for_tb

    def compute_residual(
            self,
            account_for_planet
    ):
        # 1.) move everything to the GPU
        x_train = torch.from_numpy(self.data_cube).float()

        # 2.) Get the current planet signal and subtract it if requested
        if account_for_planet:
            planet_model_idx = torch.from_numpy(np.arange(x_train.shape[0]))
            planet_signal = self.planet_model.forward(planet_model_idx)

            # 3.) Get the current data without the planet
            data_no_planet = x_train - planet_signal.squeeze().detach()
        else:
            data_no_planet = x_train

        # 4.) Set up the normalization
        x_mu = torch.mean(data_no_planet, axis=0)
        x_std = torch.std(data_no_planet, axis=0)

        # 5.) get the current normalized data
        x_norm = (x_train - x_mu) / x_std
        x_no_planet = (data_no_planet - x_mu) / x_std

        # 6.) reshape everything
        science_norm_flatten = x_norm.view(x_norm.shape[0], -1)
        science_norm_flatten_no_planet = x_no_planet.view(x_norm.shape[0], -1)

        # 7.) compute the noise estimate
        noise_estimate = self.noise_model(science_norm_flatten_no_planet)

        # 8.) compute the residual sequence
        residual_sequence = science_norm_flatten - noise_estimate
        residual_stack = residual_sequence.view(
            x_train.shape[0],
            self.noise_model.image_size,
            self.noise_model.image_size).detach().cpu().numpy()

        # 9.) Compute the unbiased median frame
        residuals_unbiased = science_norm_flatten_no_planet - noise_estimate
        residuals_unbiased = residuals_unbiased.view(
            x_train.shape[0],
            self.noise_model.image_size,
            self.noise_model.image_size).detach().cpu().numpy()

        unbiased_median_frame = np.median(residuals_unbiased, axis=0)

        # 10.) Compute the residual image
        residual_stack = residual_stack - unbiased_median_frame

        residual_after_fine_tuning = combine_residual_stack(
            residual_stack=residual_stack,
            angles=self.parang,
            combine=["Median_Residuals", ],
            suffix="",
            num_cpus=8)["Median_Residuals"]

        return residual_after_fine_tuning
