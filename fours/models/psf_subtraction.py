from typing import Union, Optional, Tuple, Dict
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


class FourS:
    """
    This is the main class that should be used to perform PSF subtraction with
    the FourS algorithm. It combines the noise model, the normalization model,
    and the rotation model. For an example of how to use this class, see the
    `Examples <../04_use_the_fours/paper_experiments/10_AF_Lep_S4_PCA.ipynb>`_.
    """

    def __init__(
            self,
            science_cube: np.ndarray,
            adi_angles: np.ndarray,
            psf_template: np.ndarray,
            noise_model_lambda: float,
            psf_fwhm: Optional[float] = None,
            right_reason_mask_factor: float = 1.5,
            rotation_grid_subsample: int = 1,
            device: Union[int, str] = 0,
            work_dir: Optional[str] = None,
            verbose: bool = True,
    ):
        """
        Initializes the FourS class for PSF subtraction.
        
        Args:
            science_cube: A 3D numpy array representing the science data cube.
                Shape: (n_frames, image_size, image_size).
            adi_angles: A 1D array of parallactic angles in radians.
            psf_template: A 2D numpy array representing the PSF template.
            noise_model_lambda: Regularization parameter for controlling the
                L2 penalty applied to the weights of the noise model. This is
                the most important hyperparameter for the noise model. Start
                with a very large value (10^8) and decrease it by a factor of
                10. Large values correspond to strong regularization (weak
                noise model, small risk of overfitting).
            psf_fwhm: Full width at half maximum (FWHM) of the PSF. If not
                provided, it is estimated using the `psf_template`.
            right_reason_mask_factor: Size of the right reason mask in units
                of the FWHM of the PSF.
            rotation_grid_subsample: Sub-sampling factor for the rotation grid
                in the ADI process. Can be used if the GPU memory is not
                sufficient for the full resolution.
            device: GPU device identifier for computation. Use 0 for the first
                GPU, "cuda:1" for the second GPU, or "cpu" for CPU computation.
            work_dir: Directory path to save models, residuals, and logs. If
                given, a tensorboard log is created. See `documentation of
                tensorboard for an example how to use it
                <https://www.tensorflow.org/tensorboard>`_.
            verbose: If True, prints progress information during the
                computation.
        """
        
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
            verbose: bool = True
    ) -> 'FourS':
        """
        Create a FourS object from a saved checkpoint. This can be used to
        restore a trained FourS model for further processing. It is possible
        to restore the noise and continue the training with a stronger
        regularization.
        
        Args:
            noise_model_file: Path to the saved noise model checkpoint.
            normalization_model_file: Path to the saved normalization model
                checkpoint, or None if no normalization is required.
            s4_work_dir: Directory for saving/loading models and residuals.
                See __init__ for more information.
            science_cube: The 3D numpy array representing the science data
                cube (frames x size x size).
            adi_angles: Array of parallactic angles in radians.
            psf_template: A 2D numpy array representing the PSF template.
            device: Device identifier for computation (e.g., "cuda:0",
                "cuda:1", or "cpu").
            verbose: If True, progress information will be printed.
        
        Returns:
            A loaded FourS instance with the restored noise and normalization
            models.
        """

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

    def _setup_work_dir(self):
        """
        Creates the directories for saving residuals, tensorboard logs, and
        models.
        
        If no work directory is given, the function returns None.
        """
        
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

    def save_models(
            self,
            file_name_noise_model: str,
            file_name_normalization_model: str
    ) -> None:
        """
        Saves the noise model and normalization model to the specified file
        paths.
        
        Args:
            file_name_noise_model: File name of the noise model to save.
            file_name_normalization_model: File name of the normalization
                model to save.
        """
        if self.verbose:
            print("Saving models ... ", end='')

        if self.models_dir is None:
            raise FileNotFoundError(
                "Saving the model requires a work directory.")

        self.normalization_model.save(
            self.models_dir / file_name_normalization_model)
        self.noise_model.save(
            self.models_dir / file_name_noise_model)

        if self.verbose:
            print("[DONE]")

    def restore_models(
            self,
            file_noise_model: Optional[str] = None,
            file_normalization_model: Optional[str] = None) -> None:
        """
        Restores the noise and normalization models from checkpoint files.
    
        Args:
            file_noise_model: Path to the saved noise model file. If None,
                the noise model is not restored.
            file_normalization_model: Path to the saved normalization model
                file. If None, the normalization model is not restored.
    
        Returns:
            None: This method does not return any value. It updates the
                `noise_model` and/or `normalization_model` in place.
        """
        if self.verbose:
            print("Restoring models ... ", end='')
    
        if file_noise_model is not None:
            self.noise_model = FourSNoise.load(file_noise_model)
    
        if file_normalization_model is not None:
            self.normalization_model = FourSFrameNormalization.load(
                file_normalization_model)

        if self.verbose:
            print("[DONE]")

    def _logg_loss_values(
            self,
            epoch: int,
            loss_recon: float,
            loss_reg: float
    ) -> None:
        """
        Logs reconstruction and regularization loss values to tensorboard.
    
        Args:
            epoch: The current epoch during training.
            loss_recon: The reconstruction loss to be logged.
            loss_reg: The regularization loss to be logged.
        """
    
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
            epoch: int,
            residual_mean: np.ndarray,
            residual_median: np.ndarray,
            training_name: str
    ) -> None:
        """
        Logs residual images to tensorboard and saves them as FITS files.
    
        Args:
            epoch: The current training epoch for which residuals are logged.
            residual_mean: The mean residual image computed during training.
            residual_median: The median residual image computed during training.
            training_name: The name of the current training session.
        """

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

    def _create_tensorboard_logger(
            self,
            training_name: str
    ) -> None:
        """
        Creates a TensorBoard logger for tracking training metrics and residuals.
    
        Args:
            training_name: A string used to name the training session. The
                directory for storing logs will append the current timestamp
                to maintain unique logging sessions.
        """
        time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        self.fine_tune_start_time = time_str
        current_logdir = self.tensorboard_dir / \
            Path(training_name + "_" + self.fine_tune_start_time)
        current_logdir.mkdir()
        self.tensorboard_logger = SummaryWriter(current_logdir)

    def _get_residual_sequence(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the residual sequence and its rotated version after applying
        the noise model and the rotation model.
    
        Returns:
            A tuple containing:
                1. rotated_residual_sequence -  The residual sequence aligned
                    to the common frame via the rotation model.
                2. residual_sequence - The raw residual sequence in its original
                    frame.
        """
        
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

    def fit_noise_model(
            self,
            num_epochs: int,
            training_name: str = "",
            logging_interval: int = 1,
            optimizer: Optional[torch.optim.Optimizer] = None,
            optimizer_kwargs: Optional[Dict] = None
    ) -> None:
        """
        Fits the noise model using the specified optimizer and training
        parameters. Call this function before calculating the residuals using
        the function `compute_residuals`.
    
        Args:
            num_epochs: The total number of epochs to train the noise model.
                The training process should converge within the number of
                epochs specified. Please check the tensorboard logs for the
                convergence of the training process. It is possible to resume
                training by calling this function again. You can even increase
                the regularization parameter to improve the noise model.
            training_name: A string to name the current training session. This
                string will be used for TensorBoard logs and residual folders.
            logging_interval: Number of epochs between logging residuals and
                losses to TensorBoard and saving residual FITS files.
            optimizer: An optional optimizer for training. If not provided,
                a default LBFGS optimizer is used.
            optimizer_kwargs: Optional arguments to initialize the optimizer.
                These include parameters such as "max_iter" and "history_size"
                for LBFGS. If the GPU memory is limited consider reducing the
                "history_size" parameter.
    
        Returns:
            None: This method performs in-place updates to the noise model
            parameters and logs the training progress or residual results.
        """
        if self.verbose:
            print("Fitting noise model ...")

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

        if self.verbose:
            print("Fitting noise model ... [DONE]")

    @torch.no_grad()
    def compute_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the final residuals for the science data by applying the 
        trained noise and rotation models to the normalized input data.

        Returns:
            A tuple containing two 2D arrays:
                1. The mean residual image created by averaging the residuals
                   in the aligned frame.
                2. The median residual image computed as the pixel-wise median
                   of the aligned residual sequence.
        """

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
