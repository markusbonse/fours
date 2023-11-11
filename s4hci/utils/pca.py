from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from s4hci.utils.logging import normalize_for_tensorboard

from s4hci.models.rotation import FieldRotationModel


def pca_psf_subtraction_gpu(
        images: np.ndarray,
        angles: np.ndarray,
        pca_numbers: np.ndarray,
        device,
        approx_svd: int,
        subsample_rotation_grid: int = 1,
        verbose: bool = False
) -> np.ndarray:
    # 1.) Convert images to torch tensor
    im_shape = images.shape
    images_torch = torch.from_numpy(images).to(device)

    # 2.) remove the mean as needed for PCA
    images_torch = images_torch - images_torch.mean(dim=0)

    # 3.) reshape images to fit for PCA
    images_torch = images_torch.view(im_shape[0], im_shape[1] * im_shape[2])

    # 4.) compute PCA basis
    if verbose:
        print("Compute PCA basis ...", end="")
    _, _, V = torch.svd_lowrank(images_torch, niter=1, q=approx_svd)
    if verbose:
        print("[DONE]")

    # 5.) build rotation model
    rotation_model = FieldRotationModel(
        all_angles=angles,
        input_size=im_shape[1],
        subsample=subsample_rotation_grid,
        inverse=False,
        register_grid=True)
    rotation_model = rotation_model.to(device)

    # 6.) compute PCA residuals for all given PCA numbers
    pca_residuals = []
    if verbose:
        print("Compute PCA residuals ...", end="")
        iter_pca_numbers = tqdm(pca_numbers)
    else:
        iter_pca_numbers = pca_numbers
    for pca_number in iter_pca_numbers:
        pca_rep = torch.matmul(images_torch, V[:, :pca_number])
        noise_estimate = torch.matmul(pca_rep, V[:, :pca_number].T)
        residual = images_torch - noise_estimate
        residual_sequence = residual.view(im_shape[0], im_shape[1], im_shape[2])

        rotated_frames = rotation_model(
            residual_sequence.unsqueeze(1).float(),
            parang_idx=torch.arange(len(residual_sequence))).squeeze(1)

        residual_torch_rot = torch.mean(rotated_frames, axis=0).cpu().numpy()
        pca_residuals.append(residual_torch_rot)

    if verbose:
        print("[DONE]")

    return np.array(pca_residuals)


def pca_tensorboard_logging(
        log_dir,
        pca_residuals,
        pca_numbers):

    summary_writer = SummaryWriter(log_dir=log_dir)

    for idx, pca_number in enumerate(pca_numbers):
        summary_writer.add_image(
            tag="Images/Residual_Mean",
            img_tensor=normalize_for_tensorboard(pca_residuals[idx]),
            global_step=pca_number,
            dataformats="HW")
