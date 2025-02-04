import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fours.utils.masks import construct_round_rfrr_template, construct_rfrr_mask


class FourSNoise(nn.Module):
    """
    This class implements the noise model for the FourS algorithm. The noise
    model consists of a linear model which is masked by a right reason mask.
    Further, it is regularized by convolving the weights with a PSF template and
    a simple L2 regularization term (lambda_reg).
    """

    def __init__(
            self,
            data_image_size: int,
            psf_template: np.ndarray,
            lambda_reg: float,
            cut_radius_psf: float,
            right_reason_mask_radius: float,
            convolve: bool=True):
        """
        Initializes the FourSNoise noise model with the given parameters.
        
        Args:
            data_image_size: The size of the image data. Only square data
                supported. (pixels)
            psf_template: A numpy array representing the point spread function
                (PSF), which is used to convolve the weights.
            lambda_reg: Regularization parameter for controlling the
                L2 penalty applied to the weights. This is the most important
                hyperparameter for the noise model. Start with a very large
                value and decrease it by a factor of 10. Large values correspond
                to strong regularization (weak noise model, small risk of
                overfitting).
            cut_radius_psf: The radius used to crop the PSF template.
            right_reason_mask_radius: Radius to construct the right reason
                mask (pixel). Should be around 1.5 times the FWHM of the PSF.
            convolve: If True, enables convolution of weights with the
                PSF template; This is the default behavior.
        """

        super(FourSNoise, self).__init__()

        # 1.) save the simple information
        self.image_size = data_image_size
        self.lambda_reg = lambda_reg
        self.convolve = convolve
        self.cut_radius_psf = cut_radius_psf
        self.right_reason_mask_radius = right_reason_mask_radius

        # 2.) prepare the psf_template
        template_cut, _ = construct_round_rfrr_template(
            radius=self.cut_radius_psf,
            psf_template_in=psf_template)

        template_norm = template_cut / np.max(np.abs(template_cut))

        self.register_buffer(
            "psf_model",
            torch.from_numpy(template_norm).unsqueeze(0).unsqueeze(0).float())

        # 3.) Initialize the raw beta values
        self.betas_raw = nn.Parameter(torch.zeros(
            self.image_size ** 2, self.image_size ** 2,
            dtype=torch.float32))

        self.prev_betas = None

        # 4.) Set up the buffers for the two masks
        right_reason_mask = construct_rfrr_mask(
            cut_off_radius=self.right_reason_mask_radius,
            psf_template_in=template_norm,
            mask_size_in=self.image_size)

        self.register_buffer(
            "right_reason_mask",
            torch.from_numpy(right_reason_mask))

    def _apply(self, fn):
        super(FourSNoise, self)._apply(fn)
        self.prev_betas = None
        return self

    def save(
            self,
            file_path: str) -> None:
        """
        Saves the noise model to a file.
        Args:
            file_path: The path to the file where the noise model should be saved.
        """

        state_dict = self.state_dict()

        # add the other information we want to keep
        state_dict["image_size"] = self.image_size
        state_dict["lambda_reg"] = self.lambda_reg
        state_dict["noise_model_convolve"] = self.convolve
        state_dict["cut_radius_psf"] = self.cut_radius_psf
        state_dict["noise_mask_radius"] = self.right_reason_mask_radius
        torch.save(state_dict, file_path)

    @classmethod
    def load(
            cls,
            file_path: str) -> 'FourSNoise':
        """
        Factory method to load a noise model from a file.

        Args:
            file_path: The path to the file where the noise model is saved.

        Returns:
            A FourSNoise object.
        """

        state_dict = torch.load(file_path)

        # create a dummy psf template
        psf_size = state_dict["psf_model"].shape[-1]
        dummy_template = np.ones((psf_size, psf_size))

        obj = cls(
            data_image_size=state_dict.pop('image_size'),
            psf_template=dummy_template,
            lambda_reg=state_dict.pop('lambda_reg'),
            cut_radius_psf=state_dict.pop('cut_radius_psf'),
            right_reason_mask_radius=state_dict.pop('noise_mask_radius'),
            convolve=state_dict.pop('noise_model_convolve'))

        obj.load_state_dict(state_dict)
        return obj

    @property
    def betas(self):
        """
        Returns the betas after applying the right reason mask and the PSF 
        template.
        """
        
        if self.prev_betas is None:
            self.compute_betas()

        return self.prev_betas

    def compute_betas(self):
        """
        Called within the forward pass to compute the betas.
        """
        
        # reshape the raw betas
        raw_betas = self.betas_raw.view(
            -1,
            self.image_size,
            self.image_size)

        # set regularization_mask values to zero
        tmp_weights = raw_betas * self.right_reason_mask

        # noise_model_convolve the weights
        if self.convolve:
            tmp_weights = F.conv2d(
                tmp_weights.unsqueeze(1),
                self.psf_model,
                padding="same").view(
                self.image_size ** 2,
                self.image_size ** 2)

        else:
            tmp_weights = tmp_weights.view(
                self.image_size ** 2,
                self.image_size ** 2)

        self.prev_betas = tmp_weights

    def forward(
            self,
            science_norm_flatten: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass to predict the noise estimate for the given 
        science data.
        
        Args:
            science_norm_flatten: Input tensor of shape
                (time, x*y), where `time` is the temporal dimension and 
                `x*y` is the flattened, already normalized image data.
        
        Returns:
            Predicted noise estimates with a shape of (time, x*y).
        """

        # we have to @beta.T because we have convolved the beta values along the
        # second axis.
        noise_estimate = science_norm_flatten @ self.betas.T

        return noise_estimate
