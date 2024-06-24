from scipy.stats import iqr

import torch
import torch.nn as nn


class FourSFrameNormalization(nn.Module):

    def __init__(
            self,
            image_size,
            normalization_type="normal"):

        super(FourSFrameNormalization, self).__init__()

        self.normalization_type = normalization_type
        if self.normalization_type not in ["normal", "robust", "dynamic"]:
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
    def load(cls, file_path):

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

        elif self.normalization_type == "robust":
            self.mean_frame = torch.median(science_data, dim=0).values
            iqr_frame = iqr(science_data.numpy(), axis=0, scale=1.349)
            self.std_frame = torch.from_numpy(iqr_frame).float()

        elif self.normalization_type == "dynamic":
            # the mean and std are calculated during the forward pass
            self.mean_frame = torch.zeros_like(science_data[0])
            self.std_frame = torch.ones_like(science_data[0])
        else:
            raise ValueError("normalization type unknown.")

    def normalize_data(
            self,
            science_data,
            re_center=True):
        if self.normalization_type == "dynamic":
            # update the mean and std
            self.mean_frame = torch.mean(science_data, axis=0)
            self.std_frame = torch.std(science_data, axis=0)

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
