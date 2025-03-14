from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fours.utils.logging import normalize_for_tensorboard
from fours.models.rotation import FieldRotationModel


def pca_psf_subtraction_gpu(
        images: np.ndarray,
        angles: np.ndarray,
        pca_numbers: np.ndarray,
        device: str,
        approx_svd: int = -1,
        subsample_rotation_grid: int = 1,
        verbose: bool = False,
        combine: str = "mean"
) -> np.ndarray:
    """
    Perform PCA-based PSF subtraction on GPU for a data set taken with 
    pupil-tracking. The PCA components are computed from the input images
    and then subtracted from the input images.
    This is a fast GPU implementation that uses PyTorch.

    Args:
        images: 3D array of shape (n_frames, height, width) representing the
            input image frames.
        angles: 1D array of parallactic angles corresponding to the image
            frames.
        pca_numbers: Array of integers specifying the number of  PCA components
            to use for reconstruction.
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        approx_svd: Number of iterations for low-rank SVD approximation
            (-1 for exact SVD). Defaults to -1.
        subsample_rotation_grid: Subsampling factor for the rotation grid.
            Defaults to 1.
        verbose: If True, print progress updates. Defaults to False.
        combine: Method to combine rotated residual images, either "mean" or
            "median". Defaults to "mean".

    Returns:
        Array of PCA-subtracted residual mean or median images, shape
        (len(pca_numbers), height, width).
    """
      
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

    if approx_svd == -1:
        _, _, V = torch.linalg.svd(images_torch)
        V = V.T
    else:
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

    for pca_number in pca_numbers:
        pca_rep = torch.matmul(images_torch, V[:, :pca_number])
        noise_estimate = torch.matmul(pca_rep, V[:, :pca_number].T)
        residual = images_torch - noise_estimate
        residual_sequence = residual.view(im_shape[0], im_shape[1], im_shape[2])

        if combine != "mean":
            residual_sequence = residual_sequence - torch.median(
                residual_sequence, axis=0)[0]

        rotated_frames = rotation_model(
            residual_sequence.unsqueeze(1).float(),
            parang_idx=torch.arange(len(residual_sequence))).squeeze(1)

        if combine == "mean":
            residual_torch_rot = torch.mean(
                rotated_frames, axis=0).cpu().numpy()
        else:
            residual_torch_rot = torch.median(
                rotated_frames, axis=0)[0].cpu().numpy()
        pca_residuals.append(residual_torch_rot)

    if verbose:
        print("[DONE]")

    return np.array(pca_residuals)


def pca_tensorboard_logging(
        log_dir: str,
        pca_residuals: np.ndarray,
        pca_numbers: np.ndarray
) -> None:
    """
    Log PCA-subtracted residual images to TensorBoard for visualization.

    Args:
        log_dir: Directory to save TensorBoard logs.
        pca_residuals: Array containing PCA residual images, of shape
            (n_components, height, width).
        pca_numbers: Array of PCA component numbers corresponding to each
            residual image.
    """
    

    summary_writer = SummaryWriter(log_dir=log_dir)

    for idx, pca_number in enumerate(pca_numbers):
        summary_writer.add_image(
            tag="Images/Residual_Mean",
            img_tensor=normalize_for_tensorboard(pca_residuals[idx]),
            global_step=int(pca_number),
            dataformats="HW")
