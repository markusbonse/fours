
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import rescale

from s4hci.models.rotation import FieldRotationModel
from s4hci.utils.masks import construct_planet_mask


class S4Planet(nn.Module):

    def __init__(
            self,
            data_image_size,
            psf_template,
            # used for the inner radius of the planet mask
            inner_mask_radius=0,
            use_up_sample=1,
            init_noise_factor=0.00001):

        super(S4Planet, self).__init__()

        # 1.) Prepare the PSF template
        psf_model_scaled = rescale(psf_template,
                                   use_up_sample,
                                   preserve_range=True)

        # the psf template is not trainable but also send to the gpu.
        self.register_buffer(
            "psf_model",
            torch.from_numpy(psf_model_scaled).unsqueeze(0).unsqueeze(0).float()
        )

        self.output_size = data_image_size
        self.input_size = int(data_image_size * use_up_sample)
        self.inner_mask_radius = inner_mask_radius

        # 2.) Init the planet model
        # values equal to zero can cause numerical instability
        self.planet_model = nn.Parameter(
            torch.abs(torch.randn(
                1,
                self.input_size,
                self.input_size).float()) * init_noise_factor,
            requires_grad=True)

        # 3.) Set up the planet mask
        planet_mask = construct_planet_mask(
            self.input_size,
            int(inner_mask_radius * use_up_sample))  # inner region mask

        # the mask is not trainable but also send to the gpu.
        self.register_buffer(
            "planet_mask",
            torch.from_numpy(planet_mask).unsqueeze(0).float())

        # 4.) There are several member variables which are set during training.
        # We initialize them here with None
        # 4.1 ) rotation grid for the training data
        self.rotation = None

    def save(self, file_path):
        state_dict = self.state_dict()

        # add the other information we want to keep
        state_dict["use_up_sample"] = self.use_up_sample
        state_dict["output_size"] = self.output_size
        state_dict["inner_mask_radius"] = self.inner_mask_radius

        torch.save(state_dict, file_path)

    @classmethod
    def load(cls, file_path):
        state_dict = torch.load(file_path)

        # create a dummy psf template
        psf_size = state_dict["psf_model"].shape[-1]
        dummy_template = np.ones((psf_size, psf_size))

        obj = cls(
            data_image_size=state_dict.pop('output_size'),
            psf_template=dummy_template,
            inner_mask_radius=state_dict.pop('inner_mask_radius'),
            use_up_sample=state_dict.pop('use_up_sample'))

        obj.load_state_dict(state_dict)

        return obj

    def setup_for_training(
            self,
            all_angles,
            rotation_grid_down_sample,
            upload_rotation_grid=False):

        # Build the rotation grid for the training data
        # we have to rotate into the opposite direction
        self.rotation = FieldRotationModel(
            all_angles,
            input_size=self.input_size,
            inverse=True,
            subsample=rotation_grid_down_sample,
            register_grid=upload_rotation_grid)

    @property
    def planet_parameters(self):
        return self.planet_model ** 2

    def get_planet_signal(self):
        planet_signal = F.conv2d(
            self.planet_parameters.unsqueeze(0),
            self.psf_model,
            padding="same")

        # mask circular pattern
        masked_planet_signal = planet_signal.squeeze(0) * self.planet_mask
        return masked_planet_signal

    def forward(self,
                parang_idx=None,
                new_angles=None):

        if new_angles is not None:
            num_copies = len(new_angles)
        else:
            num_copies = len(parang_idx)

        # 1.) Get the current planet signal and repeat it
        raw_planet_signal = self.get_planet_signal()

        # create stack of planet frames
        planet_stack = raw_planet_signal.repeat(num_copies, 1, 1).clone()

        # 2.) Rotate the planet frames
        output_dim = (self.output_size,
                      self.output_size)

        planet_stack = self.rotation(
            planet_stack.unsqueeze(1),
            parang_idx=parang_idx,
            new_angles=new_angles,
            output_dimensions=output_dim)

        return planet_stack
