import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegFC(nn.Module):

    def __init__(
            self,
            psf_template,
            all_angles,
            input_size,
            init_separation,
            init_pos_angle,
            init_flux_ratio,
            dit_science,
            dit_psf_template,
            nd_factor,
            interpolation="bicubic"
    ):

        super(NegFC, self).__init__()
        self.m_input_size = input_size

        # initialize the three parameters
        self.pos_angle = nn.Parameter(torch.Tensor([init_pos_angle]))
        self.flux_ratio = nn.Parameter(torch.Tensor([init_flux_ratio]))
        self.separation = nn.Parameter(torch.Tensor([init_separation]))
        self.interpolation = interpolation

        # the DIT and ND factor
        # TODO check if nd_factor is 1/nd_factor
        self.integration_time_factor = (
                dit_science / dit_psf_template * nd_factor)

        # save the angles
        self.register_buffer(
            "par_angles",
            torch.Tensor(all_angles,
                         dtype=torch.float32))

        # pad the psf template
        pad_size = (psf_template.shape[0] - self.m_input_size) // 2

        padded_psf = np.pad(
            psf_template,
            pad_width=((pad_size, pad_size), (pad_size, pad_size)),
            mode='constant',
            constant_values=0)

        self.register_buffer(
            "psf_template",
            torch.Tensor(padded_psf))

    def get_forward_model(self):

        # calculate the correct position angle and x/y shifts
        ang = torch.deg2rad(self.pos_angle) + torch.pi / 2 - self.par_angles

        psf_torch = (self.psf_template *
                     self.flux_ratio *
                     self.integration_time_factor)

        x_shift = self.separation * torch.cos(ang)
        y_shift = self.separation * torch.sin(ang)

        # create the affine matrix
        theta = torch.zeros(ang.shape[0], 2, 3, device=ang.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1

        # the -1 is needed because the grid uses align_corners=True
        theta[:, 0, 2] = - x_shift / (psf_torch.shape[0] - 1) * 2
        theta[:, 1, 2] = - y_shift / (psf_torch.shape[1] - 1) * 2

        # build the grid
        grid = F.affine_grid(
            theta,
            torch.Size([ang.shape[0],
                        1,
                        self.m_input_size,
                        self.m_input_size]),
            align_corners=True)

        # apply the grid
        shifted_data = F.grid_sample(
            psf_torch.unsqueeze(0).unsqueeze(0).repeat(
                ang.shape[0], 1, 1, 1),
            grid,
            mode=self.interpolation,
            align_corners=True).squeeze()

        return shifted_data

    def forward(self, science_sequence):
        # get the forward model
        forward_model = self.get_forward_model()

        # apply the forward model
        return science_sequence - forward_model

