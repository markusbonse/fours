import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldRotationModel(nn.Module):

    def __init__(self,
                 all_angles,
                 input_size,
                 subsample=1,
                 inverse=False,
                 register_grid=False):
        """
        :param all_angles: The para angles are needed in advance in order to
        save computational costs. A re-calculation of the affine grid which is
        expensive can be avoided.
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

    def _build_grid_from_angles(self,
                                angles,
                                subsample):
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

    def forward(self,
                frame_stack,
                parang_idx=None,
                new_angles=None,
                output_dimensions=None):

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
