from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp

from s4hci.utils.masks import construct_round_rfrr_template, construct_rfrr_mask
from s4hci.utils.s4_rigde import compute_betas_least_square, compute_betas_svd
from s4hci.utils.positions import get_validation_positions


class S4ClosedForm:

    def __init__(
            self,
            psf_template,
            lambda_reg,
            cut_radius_psf,
            mask_template_setup,
            convolve=True,
            use_normalization=True,
            re_mask=False,
            verbose=True,
            available_devices="cpu",
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
        self.verbose = verbose

        # 1.) save the other parameters
        self.lambda_reg = lambda_reg
        self.convolve = convolve
        self.use_normalization = use_normalization
        self.cut_radius_psf = cut_radius_psf
        self.mask_template_setup = mask_template_setup
        self.re_mask = re_mask

        # 2.) Parameters filled during training
        self.betas = None
        self.right_reason_mask = None
        self.second_mask = None
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

        self.second_mask = construct_rfrr_mask(
            template_setup=self.mask_template_setup,
            psf_template_in=self.template_norm,
            mask_size_in=self.image_size,
            use_template=True)

        if self.verbose:
            print("[DONE]")

        # 2.) Normalize the data and psf template
        if self.verbose:
            print("Build normalization frames ... ", end='')
        self.mean_frame = np.mean(science_data, axis=0)
        if self.use_normalization:
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

        # 3.) re-mask if requested
        if self.re_mask:
            second_mask = torch.Tensor(self.second_mask).view(
                self.second_mask.shape[0], -1)
            self.betas = self.betas * second_mask

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
                rank=self.available_devices[0],
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

        return all_results, best_lambda

    def _validate_lambdas_separation(
            self,
            separation,
            lambdas,
            science_data_train,
            science_data_test,
            num_test_positions,
            rank,
            approx_svd=-1
    ):

        # 1.) get the positions where we evaluate the residual error
        if self.verbose:
            print("Compute validation positions for "
                  "separation " + str(separation) + " ...")

        positions = get_validation_positions(
            num_positions=num_test_positions,
            separation=separation,
            test_image=science_data_train[0])

        # 2.) Set up the training data
        if self.verbose:
            print("Setup training data for "
                  "separation " + str(separation) + " ...")

        self._setup_training(science_data_train)
        self.science_data_norm = self.normalize_data(science_data_train)

        # 3.) Compute the betas
        # collect all parameters for the SVD
        if self.verbose:
            print("Compute betas for "
                  "separation " + str(separation) + " ...")

        X_torch = torch.from_numpy(self.science_data_norm).unsqueeze(1)
        M_torch = torch.from_numpy(self.right_reason_mask)

        if self.convolve:
            p_torch = torch.from_numpy(
                self.template_norm).unsqueeze(0).unsqueeze(0)
        else:
            p_torch = None

        betas_conv = compute_betas_svd(
            X_torch=X_torch,
            M_torch=M_torch,
            lambda_regs=lambdas,
            positions=positions,
            p_torch=p_torch,
            rank=rank,
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

        science_test = torch.from_numpy(science_data_test)
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

        # clean up
        self.science_data_norm = None

        # normalize
        median_errors = np.array(median_errors)
        median_errors -= np.mean(median_errors)
        median_errors /= np.std(median_errors)

        return median_errors

    @classmethod
    def restore_from_checkpoint(
            cls,
            checkpoint_file,
            verbose=True,
            available_devices="cpu"
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
            available_devices=available_devices)

        cls_instance.betas = checkpoint["betas"]
        cls_instance.template_norm = checkpoint["template_norm"]
        cls_instance.std_frame = checkpoint["std_frame"]
        cls_instance.mean_frame = checkpoint["mean_frame"]
        cls_instance.right_reason_mask = checkpoint["right_reason_mask"]
        cls_instance.image_size = cls_instance.mean_frame.shape[0]

        return cls_instance

    def predict(
            self,
            science_data
    ):
        science_norm = torch.from_numpy(science_data)
        science_norm = self.normalize_data(science_norm)

        # reshape the science data
        science_norm = science_norm.view(science_norm.shape[0], -1)

        # move data and weights to GPU
        science_norm = science_norm.to(self.available_devices[0])
        self.betas = self.betas.to(self.available_devices[0])

        # predict and compute residual
        noise_estimate = science_norm @ self.betas.T
        residual = science_norm - noise_estimate

        # reshape data
        noise_estimate = noise_estimate.view(
            science_norm.shape[0],
            self.image_size,
            self.image_size)

        residual = residual.view(
            science_norm.shape[0],
            self.image_size,
            self.image_size)

        # move everything back to the host
        self.betas = self.betas.cpu()
        noise_estimate = noise_estimate.cpu().numpy()
        residual = residual.cpu().numpy()

        return noise_estimate, residual