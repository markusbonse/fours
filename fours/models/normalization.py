from scipy.stats import iqr

import torch
import torch.nn as nn


class FourSFrameNormalization(nn.Module):
    """
    This class is used to normalize the science data.
    """

    def __init__(
            self,
            image_size: int,
            normalization_type: str = "normal") -> None:
        """
        Initializes the FourSFrameNormalization module.
        
        Args:
            image_size: Size of one side of the square input image, in pixels.
            normalization_type: Type of normalization to apply. "normal" for 
                mean and standard deviation, "robust" for median and IQR.
        """

        super(FourSFrameNormalization, self).__init__()

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

    def save(
            self, 
            file_path: str) -> None:
        """
        Saves the normalization parameters and metadata to the given file 
        path. The state dictionary includes the type of normalization, the 
        image size, and precomputed statistics.
    
        Args:
            file_path: The path to save the parameters and metadata.
        """
        state_dict = self.state_dict()
        state_dict["normalization_type"] = self.normalization_type
        state_dict["image_size"] = self.image_size
        torch.save(state_dict, file_path)

    @classmethod
    def load(
            cls, 
            file_path
        ) -> 'FourSFrameNormalization':
        """
        Factory method to load a normalization object from a file.
    
        Args:
            file_path: Path to the file storing the normalization parameters 
                and metadata.
    
        Returns:
            FourSFrameNormalization instance, initialized with loaded 
            parameters.
        """
    
        state_dict = torch.load(file_path)
    
        obj = cls(
            image_size=state_dict.pop("image_size"),
            normalization_type=state_dict.pop("normalization_type"))
    
        obj.load_state_dict(state_dict)
        return obj

    def prepare_normalization(
            self,
            science_data: torch.Tensor
    ) -> None:
        """
        Computes and sets the normalization parameters (mean and standard 
        deviation or robust metrics) based on the provided science data.
    
        Args:
            science_data: A 3D tensor containing science data frames used for 
                normalization. Dimensions are expected as [frames, height, 
                width].
    
        Returns:
            None. The normalization parameters are updated and stored within 
            the instance.
        """

        if self.normalization_type == "normal":
            self.mean_frame = torch.mean(science_data, axis=0)
            self.std_frame = torch.std(science_data, axis=0)

        elif self.normalization_type == "robust":
            self.mean_frame = torch.median(science_data, dim=0).values
            iqr_frame = iqr(science_data.numpy(), axis=0, scale=1.349)
            self.std_frame = torch.from_numpy(iqr_frame).float()
        else:
            raise ValueError("normalization type unknown.")

    def normalize_data(
            self,
            science_data: torch.Tensor,
            re_center: bool = True
    ) -> torch.Tensor:
        """
        Normalizes the provided science data using prepared parameters.
    
        Args:
            science_data: Input 3D tensor containing science data, with 
                dimensions [frames, height, width]. Expected to match the 
                dimensions used in `prepare_normalization`.
            re_center: Optional; if True, the mean is subtracted from the data 
                before normalization. Defaults to True.
    
        Returns:
            A tensor of normalized science data, where each value is scaled 
            by the computed standard deviation or IQR. Any NaNs are replaced 
            with 0.
        """

        if re_center:
            science_data_mean_shift = science_data - self.mean_frame
        else:
            science_data_mean_shift = science_data

        normalized_data = science_data_mean_shift / self.std_frame

        return torch.nan_to_num(normalized_data, 0)

    def forward(
            self,
            science_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward pass by normalizing the provided science data tensor.
        
        Args:
            science_data: A 3D tensor containing input science data with 
                dimensions [frames, height, width]. Data should match the 
                format used in `prepare_normalization`.
    
        Returns:
            A normalized tensor where the values are scaled by the computed 
            normalization parameters and NaNs are replaced with 0.
        """
    
        return self.normalize_data(science_data)
