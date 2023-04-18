import numpy as np

import torch
import torch.multiprocessing as mp

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask
from s4hci.utils.s4_rigde import compute_betas_least_square, compute_betas_svd


class S4Ridge:

    def __init__(
            self,
            psf_template,
            lambda_reg,
            cut_radius_psf,
            mask_template_setup,
            convolve=True,
            use_normalization=True,
            verbose=True,
            available_devices="cpu",
            half_precision=False
):
        """

        Args:
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
        self.verbose = verbose

        # 1.) save the other parameters
        self.lambda_reg = lambda_reg
        self.convolve = convolve
        self.use_normalization = use_normalization
        self.cut_radius_psf = cut_radius_psf
        self.mask_template_setup = mask_template_setup

        # 2.) Parameters filled during training
        self.betas = None
        self.right_reason_mask = None
        self.image_size = None
        self.mean_frame = None
        self.std_frame = None
        self.science_data_norm = None

        # 3.) prepare the psf_template
        template_cut, _ = construct_round_rfrr_template(
            radius=self.cut_radius_psf,
            psf_template_in=psf_template)

        self.template_norm = template_cut / np.max(np.abs(template_cut))

    def _setup_training(
            self,
            science_data):

        # 1.) Construct the right reason masks
        if self.verbose:
            print("Creating right reason mask ... ", end='')
        self.image_size = science_data.shape[1]

        self.right_reason_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=self.template_norm,
            mask_size_in=self.image_size)

        if self.verbose:
            print("[DONE]")

        # 2.) Normalize the data and psf template
        if self.verbose:
            print("Build normalization frames ... ", end='')
        self.mean_frame = np.mean(science_data, axis=0)
        if self.normalize_data:
            self.std_frame = np.std(science_data, axis=0)
        else:
            self.std_frame = np.ones_like(np.std(science_data, axis=0))

        if self.verbose:
            print("[DONE]")

    def normalize_data(self, science_data):
        science_data_mean_shift = science_data - self.mean_frame
        return science_data_mean_shift / self.std_frame

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
            lambda_reg=self.lambda_reg,
            positions=positions,
            rank=rank,
            half_precision=self.half_precision,
            verbose=self.verbose)

        return beta_conv

    def _fit_mp(
            self,
            positions
    ):

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

    def fit(
            self,
            science_data):

        self._setup_training(science_data)
        self.science_data_norm = self.normalize_data(science_data)

        positions = [(x, y)
                     for x in range(self.image_size)
                     for y in range(self.image_size)]

        if self.verbose:
            print("Fitting ... ", end='')

        # 2.) Run everything with multiprocessing
        self.betas = self._fit_mp(positions)

        if self.verbose:
            print("[DONE]")

        # clean up
        self.science_data_norm = None

    def save(
            self,
            result_file
    ):
        # create a checkpoint dict
        checkpoint = {
            "lambda_reg": self.lambda_reg,
            "betas": self.betas,
            "convolve": self.convolve,
            "use_normalization": self.use_normalization,
            "cut_radius_psf": self.cut_radius_psf,
            "mask_template_setup": self.mask_template_setup,
            "right_reason_mask": self.right_reason_mask,
            "mean_frame": self.mean_frame,
            "std_frame": self.std_frame,
            "template_norm": self.template_norm,
        }

        torch.save(checkpoint, result_file)

    @classmethod
    def restore_from_checkpoint(
            cls,
            checkpoint_file,
            verbose=True,
            available_devices="cpu",
            half_precision=False
    ):

        checkpoint = torch.load(checkpoint_file)

        cls_instance = cls(
            psf_template=checkpoint["template_norm"],
            lambda_reg=checkpoint["lambda_reg"],
            cut_radius_psf=checkpoint["cut_radius_psf"],
            mask_template_setup=checkpoint["mask_template_setup"],
            convolve=checkpoint["convolve"],
            use_normalization=checkpoint["use_normalization"],
            verbose=verbose,
            available_devices=available_devices,
            half_precision=half_precision)

        cls_instance.betas = checkpoint["betas"]
        cls_instance.template_norm = checkpoint["template_norm"]
        cls_instance.std_frame = checkpoint["std_frame"]
        cls_instance.mean_frame = checkpoint["mean_frame"]
        cls_instance.right_reason_mask = checkpoint["right_reason_mask"]
        cls_instance.image_size = cls_instance.mean_frame.shape[0]

        return cls_instance

    def predict(self):
        raise NotImplementedError()





