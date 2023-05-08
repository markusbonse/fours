from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask
from s4hci.utils.s4_rigde import compute_betas_least_square, compute_betas_svd
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
            use_normalization=True,
            re_mask=False,
            verbose=True,
            rank="cpu"):

        super(S4Noise, self).__init__()

        # 1.) save the non-model related parameters
        self.rank = rank
        self.verbose = verbose

        # 2.) save the simple information
        self.image_size = data_image_size
        self.lambda_reg = lambda_reg
        self.convolve = convolve
        self.use_normalization = use_normalization
        self.cut_radius_psf = cut_radius_psf
        self.mask_template_setup = mask_template_setup
        self.re_mask = re_mask

        # 3.) prepare the psf_template
        template_cut, _ = construct_round_rfrr_template(
            radius=self.cut_radius_psf,
            psf_template_in=psf_template)

        template_norm = template_cut / np.max(np.abs(template_cut))

        self.register_buffer(
            "psf_model",
            torch.from_numpy(template_norm).unsqueeze(0).unsqueeze(0))

        # 4.) Initialize the raw beta values
        self.betas_raw = nn.Parameter(torch.zeros(
            self.image_size**2, self.image_size**2))

        # 5.) Set up the buffers for the two masks
        if self.verbose:
            print("Creating right reason mask ... ", end='')

        right_reason_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=self.template_norm,
            mask_size_in=self.image_size)

        self.register_buffer(
            "right_reason_mask",
            torch.from_numpy(right_reason_mask))

        second_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=self.template_norm,
            mask_size_in=self.image_size,
            use_template=True)

        self.register_buffer(
            "second_mask",
            torch.from_numpy(second_mask))

        if self.verbose:
            print("[DONE]")

        # 6.) Set up the buffers for the normalization
        self.register_buffer(
            "mean_frame",
            torch.zeros(
                self.image_size,
                self.image_size))

        self.register_buffer(
            "std_frame",
            torch.zeros(
                self.image_size,
                self.image_size))

    def save(self, file_path):
        state_dict = self.state_dict()

        # add the other information we want to keep
        state_dict["image_size"] = self.image_size
        state_dict["lambda_reg"] = self.lambda_reg
        state_dict["convolve"] = self.convolve
        state_dict["cut_radius_psf"] = self.cut_radius_psf
        state_dict["mask_template_setup"] = self.mask_template_setup
        state_dict["re_mask"] = self.re_mask
        torch.save(state_dict, file_path)

    @classmethod
    def load(
            cls,
            file_path,
            verbose=True,
            available_devices="cpu"):

        state_dict = torch.load(file_path)

        # create a dummy psf template
        psf_size = state_dict["psf_model"].shape[0]
        dummy_template = np.zeros((psf_size, psf_size))

        obj = cls(
            data_image_size=state_dict.pop('image_size'),
            psf_template=dummy_template,
            lambda_reg=state_dict.pop('lambda_reg'),
            cut_radius_psf=state_dict.pop('cut_radius_psf'),
            mask_template_setup=state_dict.pop('mask_template_setup'),
            convolve=state_dict.pop('convolve'),
            re_mask=state_dict.pop('re_mask'),
            verbose=verbose,
            available_devices=available_devices)

        obj.load_state_dict(state_dict)
        return obj

    def _prepare_normalization(
            self,
            science_data):

        if self.verbose:
            print("Build normalization frames ... ", end='')

        self.mean_frame = torch.mean(science_data, axis=0)
        self.std_frame = torch.std(science_data, axis=0)

        if self.verbose:
            print("[DONE]")

    def normalize_data(self, science_data):
        science_data_mean_shift = science_data - self.mean_frame
        return science_data_mean_shift / self.std_frame

    def fit(
            self,
            science_data):

        self._prepare_normalization(science_data)
        science_data_norm = self.normalize_data(science_data)

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

        self.betas_raw.data = compute_betas_least_square(
            X_torch=science_data_norm,
            p_torch=p_torch,
            M_torch=self.right_reason_mask,
            lambda_reg=self.lambda_reg,
            positions=positions,
            verbose=self.verbose)

        if self.verbose:
            print("[DONE]")

        # TODO Check this I think it has to go in a conv fuction
        # 3.) re-mask if requested
        #if self.re_mask:
        #    second_mask = self.second_mask.view(
        #        self.second_mask.shape[0], -1)
        #    self.betas.data = self.betas.data * second_mask

    def _validate_lambdas_separation(
            self,
            separation,
            lambdas,
            science_data_train,
            science_data_test,
            num_test_positions,
            approx_svd=-1
    ):

        # 1.) get the positions where we evaluate the residual error
        if self.verbose:
            print("Compute validation positions for "
                  "separation " + str(separation) + " ...")

        positions = get_validation_positions(
            num_positions=num_test_positions,
            separation=separation,
            test_image=science_data_train[0].numpy())

        # 2.) Set up the training data
        if self.verbose:
            print("Setup training data for "
                  "separation " + str(separation) + " ...")

        self._prepare_normalization(science_data_train)
        science_data_norm = self.normalize_data(science_data_train)

        # 3.) Compute the betas
        # collect all parameters for the SVD
        if self.verbose:
            print("Compute betas for "
                  "separation " + str(separation) + " ...")

        if self.convolve:
            p_torch = self.psf_model
        else:
            p_torch = None

        betas_conv = compute_betas_svd(
            X_torch=science_data_norm,
            M_torch=self.right_reason_mask,
            lambda_regs=lambdas,
            positions=positions,
            p_torch=p_torch,
            approx_svd=approx_svd,
            verbose=self.verbose)

        # 4.) Re-mask with self.second_mask.
        # This step is needed to cut off overflow towards the identity in case
        # of small mask sizes
        if self.verbose:
            print("Re-mask betas for "
                  "separation " + str(separation) + " ...")

        re_masked = torch.zeros_like(betas_conv)
        all_idx = []
        for i, tmp_position in enumerate(positions):
            x, y = tmp_position
            tmp_idx = x * self.image_size + y
            all_idx.append(tmp_idx)

            re_masked[i] = betas_conv[i] * self.second_mask[tmp_idx]

        # 5.) Predict
        if self.verbose:
            print("Compute validation errors for "
                  "separation " + str(separation) + " ...")

        science_test = science_data_test
        science_test = self.normalize_data(science_test)

        science_test = science_test.view(science_test.shape[0], -1)
        gt_values = science_test[:, all_idx]

        median_errors = []

        for tmp_lambda_idx in tqdm(range(re_masked.shape[1])):
            tmp_beta = re_masked[:, tmp_lambda_idx]
            tmp_beta = tmp_beta.view(tmp_beta.shape[0], -1)

            tmp_prediction = science_test @ tmp_beta.T
            tmp_residual = torch.abs(gt_values - tmp_prediction)

            tmp_median_error = torch.median(tmp_residual)
            median_errors.append(tmp_median_error)

        # normalize
        median_errors = np.array(median_errors)
        median_errors -= np.mean(median_errors)
        median_errors /= np.std(median_errors)

        return median_errors

    def validate_lambdas(
            self,
            num_separations,
            lambdas,
            science_data_train,
            science_data_test,
            num_test_positions,
            approx_svd=-1
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
                approx_svd=approx_svd
            )
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

    def convolved_weights(self):
        # set regularization_mask values to zero
        tmp_weights = self.beta_raw * self.right_reason_mask

        # convolve the weights
        tmp_weights = F.conv2d(
            tmp_weights.view(-1, 1, self.image_size, self.image_size),
            self.psf_model,
            padding="same").view(self.image_size**2, self.image_size**2)

        tmp_weights = tmp_weights * self.second_mask

        return tmp_weights

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        size = x.size()
        input_flatten = x.view(-1, self.num_flat_features(x))
        convolved_weights = self.convolved_weights()
        result = F.linear(input_flatten, convolved_weights, None)
        return result.view(size)
