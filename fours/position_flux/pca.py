import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fours.models.rotation import FieldRotationModel


class PCANoiseModel(nn.Module):

    def __init__(
            self,
            angles: np.ndarray,
            image_shape: int,
            pca_number: int,
            approx_svd: int = -1
    ):
        super(PCANoiseModel, self).__init__()

        # save the simple parameters
        self.m_pca_number = pca_number
        self.m_approx_svd = approx_svd
        self.m_image_shape = image_shape

        # create the rotation model
        self.rotation_model = FieldRotationModel(
            all_angles=angles,
            input_size=self.m_image_shape,
            subsample=1,
            inverse=False,
            register_grid=True)

    def compute_basis(
            self,
            images):

        if self.m_approx_svd == -1:
            _, _, basis = torch.linalg.svd(images)
            basis = basis.T
        else:
            _, _, basis = torch.svd_lowrank(
                images, niter=1, q=self.m_approx_svd)

        return basis

    def forward(self, images):
        # 1.) reshape images to fit for PCA
        images = images.view(images.shape[0],
                             images.shape[1] * images.shape[2])

        # 2.) remove the mean as needed for PCA
        images = images - images.mean(dim=0)

        # 3.) compute PCA basis
        basis = self.compute_basis(images)

        # 4.) compute PCA residuals
        pca_rep = torch.matmul(images, basis[:, :self.m_pca_number])
        noise_estimate = torch.matmul(pca_rep, basis[:, :self.m_pca_number].T)
        residual = images - noise_estimate
        residual_sequence = residual.view(
            images.shape[0],
            self.m_image_shape,
            self.m_image_shape)

        # 5.) rotate the frames
        rotated_frames = self.rotation_model(
            residual_sequence.unsqueeze(1).float(),
            parang_idx=torch.arange(len(residual_sequence))).squeeze(1)

        # 6.) average along the time axis
        return rotated_frames.mean(dim=0)
