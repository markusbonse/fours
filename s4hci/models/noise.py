
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask


class S4Ridge:

    def __init__(
            self,
            science_data,
            psf_template,
            alpha,
            normalize_data=True,
            available_devices="cpu",
            half_precision=True,
            cut_radius_psf=4,
            mask_template_setup=("radius", 5)):
        """

        Args:
            science_data: X with shape (num_frames, y, x)
            psf_template: PSF template with shape (y, x)
            available_devices: can be "cpu" or a list of ints for gpus
            cut_radius_psf:
            mask_template_setup:
        """

        # 0.) save available devices and half_precision
        if available_devices == "cpu":
            self.available_devices = ["cpu", ]
        else:
            self.available_devices = available_devices
        self.num_devices = len(self.available_devices)
        self.half_precision = half_precision
        self.alpha = alpha

        # 1.) Construct the right reason masks
        print("Creating masks ... ", end='')
        self.image_size = science_data.shape[1]

        template_cut, _ = construct_round_rfrr_template(
            radius=cut_radius_psf,
            psf_template_in=psf_template)

        self.right_reason_mask = construct_rfrr_mask(
            template_setup=mask_template_setup,
            psf_template_in=template_cut,
            mask_size_in=self.image_size)
        print("[DONE]")

        # 2.) Normalize the data and psf template
        print("Normalizing data ... ", end='')
        self.mean_frame = np.mean(science_data, axis=0)
        self.science_data = science_data - self.mean_frame
        if normalize_data:
            self.std_frame = np.std(self.science_data, axis=0)
        else:
            self.std_frame = np.ones_like(np.std(self.science_data, axis=0))
        self.science_data_norm = self.science_data / self.std_frame

        self.template_norm = template_cut / np.max(np.abs(template_cut))

        print("[DONE]")

    def _fit(
            self,
            positions,
            rank):

        # get all the data we need as pytroch tensors
        X_torch = torch.from_numpy(self.science_data_norm).unsqueeze(1)
        p_torch = torch.from_numpy(self.template_norm).unsqueeze(0).unsqueeze(0)
        M_torch = torch.from_numpy(self.right_reason_mask)
        eye = torch.eye(self.image_size**2, self.image_size**2) * self.alpha

        if self.half_precision:
            X_torch = X_torch.float()
            p_torch = p_torch.float()
            M_torch = M_torch.float()

        # send everything to the current GPU / device
        X_torch = X_torch.to(rank)
        p_torch = p_torch.to(rank)
        M_torch = M_torch.to(rank)
        eye = eye.to(rank)

        # convolve the data
        X_conv = F.conv2d(
            X_torch, p_torch, padding="same").view(X_torch.shape[0], -1)

        X_conv_square = X_conv.T @ X_conv

        # Compute all betas in a loop over all positions
        betas = []

        for x, y in tqdm(positions):
            tmp_idx = x * self.image_size + y

            # get the current mask
            m_torch = M_torch[tmp_idx].flatten()
            Y_torch = X_torch.view(X_torch.shape[0], -1)[:, tmp_idx]

            # compute MPX^TXPM + lambda * eye
            matrix_within = ((X_conv_square * m_torch).T * m_torch).T + eye

            # compute beta
            beta = torch.linalg.inv(matrix_within) @ (
                        X_conv * m_torch).T @ Y_torch
            beta_cut = (beta * m_torch).cpu()

            betas.append(beta_cut.cpu())

        # Convolve the results
        betas_conv = torch.stack(betas).reshape(
            len(betas), 1, self.image_size, self.image_size).to(rank)

        beta_conv = F.conv2d(
            betas_conv, p_torch, padding="same")

        beta_conv = beta_conv.squeeze().cpu()

        return beta_conv

    def fit_and_validate(
            self,
            step_size,
            test_science_data):

        # 1.) Compute a grid of positions to run the validation on
        test_positions = [(y, x) for x in range(0, self.image_size, step_size)
                          for y in range(0, self.image_size, step_size)]

        # 2.) Run everything with multiprocessing
        position_splits = np.array_split(test_positions, self.num_devices)

        experiments = list(zip(position_splits,
                               self.available_devices))

        mp.set_start_method("spawn", force=True)

        pool = mp.Pool(processes=self.num_devices)
        results = pool.starmap(self._fit, experiments)
        pool.close()
        pool.join()

        # 3.) collect and betas from the mp results
        tmp_betas = torch.cat(results, dim=0).flatten(start_dim=1)

        # 4.) prepare the test data for pytorch
        # Normalize the test data
        X_test_norm = test_science_data - self.mean_frame
        X_test_norm = X_test_norm / self.std_frame
        X_test_torch = torch.from_numpy(X_test_norm)
        X_test_torch = X_test_torch.flatten(start_dim=1).float()

        # 5.) make the prediction
        Y_test = X_test_torch.float() @ tmp_betas.T

        # 6.) Collect the true Y from X_test_torch and compute the errors
        idx_positions = [x * self.image_size + y for x, y in test_positions]
        abs_errors = torch.abs(Y_test - X_test_torch[:, idx_positions])

        return abs_errors.numpy()

    def predict(self):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()





