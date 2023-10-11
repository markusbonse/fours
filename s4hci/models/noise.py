import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask
from s4hci.utils.s4_rigde import compute_betas, compute_betas_svd
from s4hci.utils.positions import get_validation_positions


class S4Noise(nn.Module):

    def __init__(
            self,
            data_image_size,
            psf_template,
            lambda_reg,
            cut_radius_psf,
            mask_template_setup,
            convolve=True,
            verbose=True):

        super(S4Noise, self).__init__()

        # 1.) save the non-model related parameters
        self.verbose = verbose

        # 2.) save the simple information
        self.image_size = data_image_size
        self.lambda_reg = lambda_reg
        self.convolve = convolve
        self.cut_radius_psf = cut_radius_psf
        self.mask_template_setup = mask_template_setup

        # 3.) prepare the psf_template
        template_cut, _ = construct_round_rfrr_template(
            radius=self.cut_radius_psf,
            psf_template_in=psf_template)

        template_norm = template_cut / np.max(np.abs(template_cut))

        self.register_buffer(
            "psf_model",
            torch.from_numpy(template_norm).unsqueeze(0).unsqueeze(0).float())

        # 4.) Initialize the raw beta values
        self.betas_raw = nn.Parameter(torch.zeros(
            self.image_size ** 2, self.image_size ** 2,
            dtype=torch.float32))

        self.prev_betas = None

        # 5.) Set up the buffers for the two masks
        right_reason_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=template_norm,
            mask_size_in=self.image_size)

        self.register_buffer(
            "right_reason_mask",
            torch.from_numpy(right_reason_mask))

    def _apply(self, fn):
        super(S4Noise, self)._apply(fn)
        self.prev_betas = None
        return self

    @staticmethod
    def _print_progress(msg):
        def decorator(function):
            def wrapper(self, *args, **kwargs):
                if self.verbose:
                    print(msg + " ... ", end='')
                    function(*args, **kwargs)
                    print("[DONE]")
                else:
                    function(*args, **kwargs)
            return wrapper
        return decorator

    @_print_progress("S4 Noise: saving noise model")
    def save(self, file_path):
        state_dict = self.state_dict()

        # add the other information we want to keep
        state_dict["image_size"] = self.image_size
        state_dict["lambda_reg"] = self.lambda_reg
        state_dict["convolve"] = self.convolve
        state_dict["cut_radius_psf"] = self.cut_radius_psf
        state_dict["mask_template_setup"] = self.mask_template_setup
        torch.save(state_dict, file_path)

    @classmethod
    def load(
            cls,
            file_path,
            verbose=True):

        state_dict = torch.load(file_path)

        # create a dummy psf template
        psf_size = state_dict["psf_model"].shape[-1]
        dummy_template = np.ones((psf_size, psf_size))

        obj = cls(
            data_image_size=state_dict.pop('image_size'),
            psf_template=dummy_template,
            lambda_reg=state_dict.pop('lambda_reg'),
            cut_radius_psf=state_dict.pop('cut_radius_psf'),
            mask_template_setup=state_dict.pop('mask_template_setup'),
            convolve=state_dict.pop('convolve'),
            verbose=verbose)

        obj.load_state_dict(state_dict)
        return obj

    @_print_progress("S4 Noise: fitting noise model")
    def fit(
            self,
            science_data,
            device="cpu",
            fp_precision="float32"):

        positions = [(x, y)
                     for x in range(self.image_size)
                     for y in range(self.image_size)]

        if self.verbose:
            print("Fitting ... ", end='')

        # 2.) Fit all positions
        if self.convolve:
            p_torch = self.psf_model
        else:
            p_torch = None

        self.betas_raw.data = compute_betas(
            X_torch=science_data,
            p_torch=p_torch,
            M_torch=self.right_reason_mask,
            lambda_reg=self.lambda_reg,
            positions=positions,
            verbose=self.verbose,
            device=device,
            fp_precision=fp_precision)

        if self.verbose:
            print("[DONE]")

    def _validate_lambdas_separation(
            self,
            separation,
            lambdas,
            science_data_train,
            science_data_test,
            num_test_positions,
            approx_svd=-1,
            device="cpu"):

        # 1.) get the positions where we evaluate the residual error
        if self.verbose:
            print("Compute validation positions for "
                  "separation " + str(separation) + " ...")

        positions = get_validation_positions(
            num_positions=num_test_positions,
            separation=separation,
            test_image=science_data_train[0].cpu().numpy())

        # 2.) Compute the betas
        # collect all parameters for the SVD
        if self.verbose:
            print("Compute betas for "
                  "separation " + str(separation) + " ...")

        if self.convolve:
            p_torch = self.psf_model
        else:
            p_torch = None

        betas_final = compute_betas_svd(
            X_torch=science_data_train,
            M_torch=self.right_reason_mask,
            lambda_regs=lambdas,
            positions=positions,
            p_torch=p_torch,
            approx_svd=approx_svd,
            verbose=self.verbose,
            device=device)

        # 3.) Re-mask with second_mask.
        # This step is needed to cut off overflow towards the identity in case
        # of small mask sizes

        second_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=self.psf_model.cpu().numpy()[0, 0],
            mask_size_in=self.image_size,
            use_template=True)

        second_mask = torch.from_numpy(second_mask)

        if self.verbose:
            print("Re-mask betas for "
                  "separation " + str(separation) + " ...")

        re_masked = torch.zeros_like(betas_final)
        all_idx = []
        for i, tmp_position in enumerate(positions):
            x, y = tmp_position
            tmp_idx = x * self.image_size + y
            all_idx.append(tmp_idx)

            re_masked[i] = betas_final[i] * second_mask[tmp_idx]

        # 4.) Predict
        if self.verbose:
            print("Compute validation errors for "
                  "separation " + str(separation) + " ...")

        science_data_test = science_data_test.view(
            science_data_test.shape[0], -1)
        gt_values = science_data_test[:, all_idx]

        # move to GPU
        gt_values = gt_values.to(device)
        re_masked = re_masked.to(device)
        science_data_test = science_data_test.to(device)

        median_errors = []

        for tmp_lambda_idx in range(re_masked.shape[1]):
            tmp_beta = re_masked[:, tmp_lambda_idx]
            tmp_beta = tmp_beta.view(tmp_beta.shape[0], -1)

            tmp_prediction = science_data_test @ tmp_beta.T
            tmp_residual = torch.abs(gt_values - tmp_prediction).cpu()

            tmp_median_error = torch.median(tmp_residual)
            median_errors.append(tmp_median_error)

        # clean up memory
        del gt_values, re_masked, science_data_test

        # normalize
        median_errors = np.array(median_errors)
        median_errors -= np.mean(median_errors)
        median_errors /= np.std(median_errors)

        return median_errors

    @_print_progress("S4 Noise: validating noise model")
    def validate_lambdas(
            self,
            num_separations,
            lambdas,
            science_data_train,
            science_data_test,
            num_test_positions,
            approx_svd=-1,
            device="cpu"
    ):
        test_image = science_data_train[0]
        image_size_radius = int((test_image.shape[0] - 1) / 2)

        separations = np.linspace(0, image_size_radius, num_separations + 1,
                                  endpoint=False)[1:]

        all_results = dict()
        for tmp_separation in separations:
            tmp_errors = self._validate_lambdas_separation(
                separation=tmp_separation,
                lambdas=lambdas,
                science_data_train=science_data_train,
                science_data_test=science_data_test,
                num_test_positions=num_test_positions,
                approx_svd=approx_svd,
                device=device)

            all_results[tmp_separation] = tmp_errors

        # find the best lambda
        merged_results = np.array([i for i in all_results.values()])
        median_result = np.median(merged_results, axis=0)

        best_lambda_idx = np.argmin(median_result)
        best_lambda = lambdas[best_lambda_idx]

        if self.verbose:
            print("Recommended Lambda = {:.2f}".format(best_lambda))
            print("Make sure to check if the tested range of lambda values is "
                  "covering the global minimum!")

        self.lambda_reg = best_lambda

        return all_results, best_lambda

    @property
    def betas(self):
        if self.prev_betas is None:
            self.compute_betas()

        return self.prev_betas

    def compute_betas(self):
        # reshape the raw betas
        raw_betas = self.betas_raw.view(
            -1,
            self.image_size,
            self.image_size)

        # set regularization_mask values to zero
        tmp_weights = raw_betas * self.right_reason_mask

        # convolve the weights
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

    def predict_noise(
            self,
            science_data_norm
    ):
        """
        science_data: shape: (time, x, y), normalized raw data
        """

        # 1.) predict the noise
        with torch.no_grad():
            science_norm_flatten = science_data_norm.view(
                science_data_norm.shape[0], -1)

            self.compute_betas()
            noise_estimate = self.forward(science_norm_flatten)

        # 2.) reshape the result
        noise_estimate = noise_estimate.view(
            science_data_norm.shape[0],
            self.image_size,
            self.image_size)

        return noise_estimate

    def forward(
            self,
            science_norm_flatten: torch.Tensor
    ) -> torch.Tensor:
        """
        science_norm_flatten: shape: (time, x*y) already normalized
        """

        noise_estimate = science_norm_flatten @ self.betas.T

        return noise_estimate
