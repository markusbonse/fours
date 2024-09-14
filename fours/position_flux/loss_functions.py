import torch
import torch.nn.functional as F


from fours.utils.masks import create_aperture_mask


# code needed for the calculation of the hessian
def gaussian_kernel1d(sigma, order, radius):
    """Create a 1D Gaussian kernel of a given order and sigma."""
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    if order == 0:
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    elif order == 1:
        kernel = -x / sigma ** 2 * torch.exp(-0.5 * (x / sigma) ** 2)
    elif order == 2:
        kernel = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * torch.exp(
            -0.5 * (x / sigma) ** 2)
    else:
        raise ValueError('Order must be 0, 1, or 2')
    kernel /= kernel.abs().sum()
    return kernel


def gaussian_kernel2d(sigma, order_x, order_y, radius):
    """Create a 2D Gaussian kernel by computing the outer product of two 1D Gaussian kernels."""
    kernel_x = gaussian_kernel1d(sigma, order_x, radius)
    kernel_y = gaussian_kernel1d(sigma, order_y, radius)
    kernel_2d = torch.outer(kernel_y, kernel_x)
    return kernel_2d


def hessian_matrix(
        image,
        hxx_kernel,
        hxy_kernel,
        hyy_kernel):

    # Reshape kernels for convolution
    hxx_kernel = hxx_kernel.view(1, 1, *hxx_kernel.shape)
    hxy_kernel = hxy_kernel.view(1, 1, *hxy_kernel.shape)
    hyy_kernel = hyy_kernel.view(1, 1, *hyy_kernel.shape)

    # Add batch and channel dimensions to the image
    image = image.unsqueeze(0).unsqueeze(0)

    # Convolve the image with the kernels
    padding = (hxx_kernel.shape[-1] - 1) // 2
    hxx = F.conv2d(image, hxx_kernel, padding=padding)
    hxy = F.conv2d(image, hxy_kernel, padding=padding)
    hyy = F.conv2d(image, hyy_kernel, padding=padding)

    # Remove the batch and channel dimensions
    hxx = hxx.squeeze(0).squeeze(0)
    hxy = hxy.squeeze(0).squeeze(0)
    hyy = hyy.squeeze(0).squeeze(0)

    return hxx, hxy, hyy


class NegFCLoss(torch.nn.Module):

    def __init__(
            self,
            residual_shape,
            separation_pixel,
            pos_angle_deg,
            aperture_radius,
            metric_function="mse",
            sigma_hessian=1.8
    ):
        super(NegFCLoss, self).__init__()

        # Create the mask and transform it to a tensor
        self.loss_mask = create_aperture_mask(
            image_shape=residual_shape,
            separation_pixel=separation_pixel,
            pos_angle_deg=pos_angle_deg,
            aperture_radius=aperture_radius)

        # define the metric function
        self.metric_function = metric_function

        # if the metric function is hessian create the kernels
        if self.metric_function == "hessian":
            radius = int(3 * sigma_hessian + 0.5)
            hxx_kernel = gaussian_kernel2d(
                sigma_hessian, 2, 0, radius)
            hxy_kernel = gaussian_kernel2d(
                sigma_hessian, 1, 1, radius)
            hyy_kernel = gaussian_kernel2d(
                sigma_hessian, 0, 2, radius)

            # register the kernels
            self.register_buffer("hxx_kernel", hxx_kernel)
            self.register_buffer("hxy_kernel", hxy_kernel)
            self.register_buffer("hyy_kernel", hyy_kernel)

    def forward(self, residual_image):
        if self.metric_function == "hessian":
            hxx, hxy, hyy = hessian_matrix(
                residual_image,
                self.hxx_kernel,
                self.hxy_kernel,
                self.hyy_kernel)

            # calculate the determinant of the hessian
            hes_det = (hxx * hyy) - (hxy * hxy)

            # apply the mask
            selected_pixel = hes_det[self.loss_mask]

        elif self.metric_function == "mse":
            # apply the mask
            selected_pixel = residual_image[self.loss_mask]

        else:
            raise ValueError("The metric function is not implemented.")

        # calculate the loss
        loss = torch.sum(selected_pixel ** 2)

        return loss
