import multiprocessing

import numpy as np
import torch
from scipy.ndimage import rotate
from tqdm import tqdm

from fours.models.rotation import FieldRotationModel


def cadi_psf_subtraction(
        images: np.ndarray,
        angles: np.ndarray
) -> np.ndarray:
    """
    Perform Classical Angular Differential Imaging (cADI) PSF subtraction to 
    suppress stellar light from astronomical images.
    
    CADI is a widely acclaimed technique for high-contrast imaging that 
    suppresses point spread function (PSF) noise caused by the central star.
    The method works by creating a median frame from the input images,
    subtracting the median to create a residual image sequence, and
    derotating the residual images using the corresponding parallactic angles.
    The derotated residual stack is then combined using the mean 
    to generate the final high-contrast image.
    
    Args:
        images: A 3D NumPy array of shape (N, H, W) representing the input image
            stack. Here, N is the number of images, and H and W are the height
            and width of each image.

        angles: A 1D NumPy array of length N containing the parallactic angles
            corresponding to each input image in radians. 
    
    Returns:
        A 2D NumPy array of shape (H, W) representing the final high-contrast
        residual image after PSF subtraction.
    """

    median_frame = np.median(images, axis=0)
    residual_sequence = images - median_frame

    residual_image = combine_residual_stack(
        residual_stack=residual_sequence,
        angles=angles,
        combine="mean")

    return residual_image


def cadi_psf_subtraction_gpu(
        device: str,
        images: np.ndarray,
        angles: np.ndarray
) -> np.ndarray:
    """
    Perform GPU-accelerated Classical Angular Differential Imaging (cADI) 
    PSF subtraction to suppress stellar light from astronomical images.
    
    This function leverages PyTorch to execute computations on GPU hardware,
    enhancing speed for large datasets. It calculates and subtracts the 
    median frame, derotates residual frames using parallactic angles, and 
    combines them to create a high-contrast image.
    
    Args:
        device: The PyTorch device indicating the computation hardware
            (e.g., 'cuda' for GPU or 'cpu').
    
        images: A 3D NumPy array of shape (N, H, W) representing the input image
            sequence, where N is the number of images, and H and W are the
            height and width of each image.

        angles: A 1D NumPy array of length N containing the parallactic angles in
            radians, used for de-rotation of the residual frames.
    
    Returns:
        A 2D NumPy array of shape (H, W) representing the final high-contrast
        residual image computed after PSF subtraction and derotation.
    """

    images = torch.from_numpy(images).to(device)

    median_frame = torch.median(images, axis=0)[0]
    residual_sequence = images - median_frame

    # combine residual stack on GPU
    rotation_model = FieldRotationModel(
        all_angles=angles,
        input_size=images.shape[1],
        subsample=1,
        inverse=False,
        register_grid=True)
    rotation_model = rotation_model.to(device)

    rotated_frames = rotation_model(
        residual_sequence.unsqueeze(1).float(),
        parang_idx=torch.arange(len(residual_sequence))).squeeze(1)

    residual_image = torch.mean(rotated_frames, axis=0).cpu().numpy()

    return residual_image


def combine_residual_stack(
        residual_stack: np.ndarray,
        angles: np.ndarray,
        combine:str ="mean",
        subtract_temporal_average: bool =False,
        num_cpus:int = 4
) -> np.ndarray:
    """
    Derotate and combine the residual image stack using parallactic angles 
    to suppress speckle noise and generate the final high-contrast image.

    This function processes a sequence of residual images by spatially derotating 
    each frame based on its parallactic angle and then combines the derotated 
    stack into a single output frame. The processing pipeline includes an optional 
    subtraction of the temporal average from each residual frame, configurable 
    combination methods (mean or median), and multi-processing support.

    Args:
        residual_stack: A 3D NumPy array of shape (N, H, W) representing residual
            images from which PSF noise is to be suppressed. N is the number of
            images, H and W are the height and width of the images, respectively.

        angles: A 1D NumPy array of length N containing parallactic angles,
            measured in radians, corresponding to each image in the residual stack.
            These angles dictate the derotation applied to each frame.


        combine: Specifies the method to combine the derotated stack. Accepted 
            values are:

                1. "mean" - Compute the arithmetic mean of derotated frames.

                2. "median" - Compute the median frame from the stack.

        subtract_temporal_average: If set to True, the temporal average image
            across all frames (computed using either the mean or median) is
            subtracted from each residual image before derotation. This step helps
            mitigate temporal contamination.

        num_cpus: The number of CPU cores to utilize for parallel processing using
            Python's `multiprocessing.Pool`. If `num_cpus` is set to 1, sequential
            processing will be executed. For multi-core processing, each frame
            is rotated in parallel.

    Returns:
        A 2D NumPy array of shape (H, W) representing the final
        high-contrast image obtained after derotation and combination of the
        residual frames.

    """

    if combine == "mean":
        temporal_average = np.mean(residual_stack, axis=0)
    else:
        temporal_average = np.median(residual_stack, axis=0)

    if subtract_temporal_average:
        residual_stack = residual_stack - temporal_average

    if num_cpus == 1:
        de_rotated = []

        for i in tqdm(range(residual_stack.shape[0])):
            tmp_frame = rotate(
                residual_stack[i, :, :],
                -np.rad2deg(angles[i]),
                (0, 1),
                False)
            de_rotated.append(tmp_frame)

    else:
        arguments = [(residual_stack[i, :, :],
                      -np.rad2deg(angles[i]),
                      (0, 1),
                      False) for i in range(residual_stack.shape[0])]

        pool = multiprocessing.Pool(num_cpus)
        de_rotated = np.array(pool.starmap(rotate, arguments))
        pool.close()

    if combine == "mean":
        return np.mean(de_rotated, axis=0)
    else:
        return np.median(de_rotated, axis=0)
