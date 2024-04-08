from scipy.stats import iqr

import torch
import torch.nn as nn

from s4hci.models.rotation import FieldRotationModel


def get_radial_normalization(mean_frame):
    torch_mean_frame = torch.tensor(mean_frame).unsqueeze(0).unsqueeze(0)

    tmp_rotation_model = FieldRotationModel(
        all_angles=torch.deg2rad(torch.linspace(0, 360, 720)),
        input_size=torch_mean_frame.shape[-1],
        subsample=1,
        inverse=False,
        register_grid=False)

    # repeat the mean frame 360 time
    torch_mean_frame_sequence = torch_mean_frame.repeat(720, 1, 1, 1)

    # rotate the mean frame
    rotated_mean_frame = tmp_rotation_model(
        torch_mean_frame_sequence.float(),
        parang_idx=torch.arange(720))

    rotated_median_frame = torch.median(rotated_mean_frame, dim=0)[0].squeeze()

    return rotated_median_frame


class S4FrameNormalization(nn.Module):

    def __init__(
            self,
            image_size,
            normalization_type="normal"):

        super(S4FrameNormalization, self).__init__()

        self.normalization_type = normalization_type
        if self.normalization_type not in ["normal", "robust"]:
            raise ValueError("normalization type unknown.")

        self.register_buffer(
            "std_frame",
            torch.zeros((image_size, image_size)))
        self.register_buffer(
            "mean_frame",
            torch.zeros((image_size, image_size)))

    @property
    def image_size(self):
        return self.mean_frame.shape[0]

    def save(self, file_path):
        state_dict = self.state_dict()
        state_dict["normalization_type"] = self.normalization_type
        state_dict["image_size"] = self.image_size
        torch.save(state_dict, file_path)

    @classmethod
    def load(
            cls,
            file_path):

        state_dict = torch.load(file_path)

        obj = cls(
            image_size=state_dict.pop("image_size"),
            normalization_type=state_dict.pop("normalization_type"))

        obj.load_state_dict(state_dict)
        return obj

    def prepare_normalization(
            self,
            science_data):

        if self.normalization_type == "normal":
            self.mean_frame = torch.mean(science_data, axis=0)
            self.std_frame = torch.std(science_data, axis=0)
        elif self.normalization_type == "ring":
            self.mean_frame = get_radial_normalization(
                torch.mean(science_data, axis=0).numpy())
            self.std_frame = torch.std(science_data, axis=0)
        else:
            self.mean_frame = torch.median(science_data, dim=0).values
            iqr_frame = iqr(science_data.numpy(), axis=0, scale=1.349)
            self.std_frame = torch.from_numpy(iqr_frame).float()

    def normalize_data(
            self,
            science_data,
            re_center=True):

        if re_center:
            science_data_mean_shift = science_data - self.mean_frame
        else:
            science_data_mean_shift = science_data

        normalized_data = science_data_mean_shift / self.std_frame

        return torch.nan_to_num(normalized_data, 0)

    def forward(
            self,
            science_data):

        return self.normalize_data(science_data)
