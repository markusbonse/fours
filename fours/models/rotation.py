from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldRotationModel(nn.Module):
    """
    This class implements a field rotation model which rotates the input data
    by a sequence of angles. This model is needed to back-propagate through
    the rotation as needed for the 4S loss function.
    """

    def __init__(
            self,
            all_angles: np.ndarray,
            input_size: int,
            subsample: int = 1,
            inverse: bool = False,
            register_grid: bool = False
    ) -> None:
        """
        Initializes the FieldRotationModel with the given parameters.

        Args:
            all_angles: Predefined rotation angles used to create the affine 
                grid. This avoids expensive grid recalculation and improves 
                computational efficiency.
            input_size: The size of the input frames, assumed to have equal 
                height and width.
            subsample: Reduces the number of angles by sampling every nth 
                angle based on this value. Defaults to 1 for no subsampling.
                This is useful for speeding up the forward pass and reduce
                GPU memory consumption.
            inverse: If True, applies the inverse rotation using negative 
                angles. Defaults to False.
            register_grid: Specifies whether to precompute and register the 
                affine grid as a buffer. If it is set as a buffer, the full grid
                will be moved to the GPU during the forward pass.
                Defaults to False.
        """

        super(FieldRotationModel, self).__init__()
        self.m_input_size = input_size
        self.m_inverse = inverse
        self.m_subsample = subsample
        self.m_register_grid = register_grid

        if register_grid:
            self.register_buffer(
                "m_grid",
                self._build_grid_from_angles(
                    all_angles, self.m_subsample))
        else:
            self.m_grid = self._build_grid_from_angles(
                all_angles, self.m_subsample)

    def _build_grid_from_angles(
            self,
            angles: np.ndarray,
            subsample: int = 1
    ) -> torch.Tensor:
        """
        Builds an affine grid from a sequence of angles. The grid is used to
        rotate the input data by specified angles, either forward or inverse 
        based on the model's settings.
    
        Args:
            angles: Sequence of rotation angles in degrees.
            subsample: Sampling rate for reducing the number of angles. A value
                of 1 uses all angles, while higher values skip every nth angle.
    
        Returns:
            A torch.Tensor containing the affine grid used for data rotation.
        """

        # subsample angles if needed
        if subsample != 1:
            sub_angles = angles[::subsample]
        else:
            sub_angles = angles

        if self.m_inverse:
            angles_rad = - torch.Tensor(sub_angles).flatten()
        else:
            angles_rad = torch.Tensor(sub_angles).flatten()

        theta = torch.zeros(angles_rad.shape[0], 2, 3)

        theta[:, 0, 0] = torch.cos(angles_rad)
        theta[:, 1, 0] = -torch.sin(angles_rad)
        theta[:, 0, 1] = torch.sin(angles_rad)
        theta[:, 1, 1] = torch.cos(angles_rad)

        grid = F.affine_grid(
            theta,
            torch.Size((len(sub_angles), 1,
                        self.m_input_size,
                        self.m_input_size)),
            align_corners=True)
        return grid

    def forward(
            self,
            frame_stack: torch.Tensor,
            parang_idx: Optional[torch.Tensor] = None,
            new_angles: Optional[torch.Tensor] = None,
            output_dimensions=None
    ) -> torch.Tensor:
        """
        Rotates the input data by the specified angles using the predefined
        affine grid. The rotation is performed either using `parang_idx` or
        `new_angles`, which are mutually exclusive.
        
        Args:
            frame_stack: Batch of input frames to be rotated, of shape 
                (batch_size, channels, height, width).
            parang_idx: Indices to select precomputed affine grids for the 
                rotation. Required if `new_angles` is not specified.
            new_angles: Custom angles in degrees to build new affine grids for
                rotation. Expensive operation, used for test data or finer 
                control. Cannot be used with `parang_idx`.
            output_dimensions: Target size for up/downscaling the rotated 
                frames, specified as a tuple (height, width). Defaults to the 
                model's input size.
        
        Returns:
            Rotated tensor of shape (batch_size, channels, target_height, 
            target_width), with frames rotated by given or indexed angles.
        """

        if parang_idx is None and new_angles is None:
            raise ValueError("Either parang_idx or new_angles needs to be "
                             "different from None.")

        # 1.) In case idx are given we can use the pre-calculated grid
        if new_angles is None:
            if self.m_subsample != 1:
                sub_sampled_idx = torch.div(parang_idx.flatten(),
                                            self.m_subsample,
                                            rounding_mode='floor')
                tmp_grid = self.m_grid[sub_sampled_idx]
            else:
                tmp_grid = self.m_grid[parang_idx.flatten()]
        else:
            # In case new angles are given we need to build a new sampler grid
            # This is expensive and should only be used for test data
            tmp_grid = self._build_grid_from_angles(new_angles, subsample=1)

        if not self.m_register_grid or new_angles is not None:
            tmp_grid = tmp_grid.to(frame_stack.device)

        with torch.backends.cudnn.flags(enabled=False):
            rotated_data = F.grid_sample(frame_stack,
                                         tmp_grid,
                                         align_corners=True)

        if output_dimensions is None:
            output_dimensions = (self.m_input_size,
                                 self.m_input_size)

        rotated_data = F.interpolate(rotated_data,
                                     output_dimensions,
                                     mode="bicubic",
                                     align_corners=True)

        return rotated_data
