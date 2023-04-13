import numpy as np

import torch
import torch.multiprocessing as mp

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask
from s4hci.utils.s4_rigde import compute_betas_least_square, compute_betas_svd


class S4Ridge:

    def __init__(
            self,
            science_data,
            psf_template,
            alpha,
            verbose=True,
            convolve=True,
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
        self.betas = None
        self.verbose = verbose
        self.convolve = convolve

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
        M_torch = torch.from_numpy(self.right_reason_mask)

        if self.convolve:
            p_torch = torch.from_numpy(
                self.template_norm).unsqueeze(0).unsqueeze(0)
        else:
            p_torch = None

        beta_conv = compute_betas_least_square(
            X_torch=X_torch,
            p_torch=p_torch,
            M_torch=M_torch,
            alpha=self.alpha,
            positions=positions,
            rank=rank,
            half_precision=self.half_precision,
            verbose=self.verbose)

        return beta_conv

    def _fit_mp(
            self,
            positions):

        # 2.) Run everything with multiprocessing
        position_splits = np.array_split(positions, self.num_devices)

        experiments = list(zip(position_splits,
                               self.available_devices))

        mp.set_start_method("spawn", force=True)

        pool = mp.Pool(processes=self.num_devices)
        results = pool.starmap(self._fit, experiments)
        pool.close()
        pool.join()

        # 3.) collect and betas from the mp results
        return torch.cat(results, dim=0).flatten(start_dim=1)

    def fit(self):
        positions = [(y, x)
                     for x in range(self.image_size)
                     for y in range(self.image_size)]

        # 2.) Run everything with multiprocessing
        self.betas = self._fit_mp(positions)

    def predict(self):
        raise NotImplementedError()





