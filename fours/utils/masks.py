import numpy as np
from scipy.ndimage import shift
from photutils.aperture import CircularAperture


def construct_round_rfrr_template(
        radius,
        psf_template_in):

    if radius == 0:
        template_mask = np.zeros_like(psf_template_in)
    else:
        image_center = (psf_template_in.shape[0] - 1) / 2.

        aperture = CircularAperture(positions=(image_center, image_center),
                                    r=radius)
        template_mask = aperture.to_mask().to_image(psf_template_in.shape)
    template = psf_template_in * template_mask

    return template, template_mask


def construct_rfrr_mask(cut_off_radius,
                        psf_template_in,
                        mask_size_in: int,
                        use_template=False):

    # 1.) Create the template
    template, template_mask = construct_round_rfrr_template(
        cut_off_radius,
        psf_template_in)

    if use_template:
        template_mask = template

    # 2.) Create the mask stack by shifting the template
    regularization_mask = np.zeros((mask_size_in, mask_size_in,
                                    mask_size_in, mask_size_in))

    # 3.) shift the template_mask to create the mask
    # if the template_mask is smaller than the final mask size we pad it
    if template_mask.shape[0] < mask_size_in:
        pad_size = int((mask_size_in - template_mask.shape[0]) / 2)
        padded_template = np.pad(template_mask, pad_size)
    else:
        padded_template = template_mask

    center_offset = int((mask_size_in - 1) / 2)

    for i in range(mask_size_in):
        for j in range(mask_size_in):
            shifted_image = shift(padded_template,
                                  (float(i) - center_offset,
                                   float(j) - center_offset),
                                  order=1, mode="nearest")

            if template_mask.shape[0] < mask_size_in:
                regularization_mask[i, j] = shifted_image
            else:
                template_cut = int((template_mask.shape[0] - mask_size_in) / 2)
                regularization_mask[i, j] = \
                    shifted_image[
                    template_cut:-template_cut,
                    template_cut:-template_cut]

    regularization_mask = regularization_mask.reshape(
        (-1, mask_size_in, mask_size_in))

    # invert the mask
    regularization_mask = regularization_mask.astype(np.float32)
    regularization_mask = np.abs(regularization_mask - 1)

    return regularization_mask


def create_aperture_mask(
        image_shape,
        separation_pixel,
        pos_angle_deg,
        aperture_radius):
    """
    This function is needed to mask the area around a potential detection.
    """

    # calculate the x, y position of the mask
    image_center = np.array(image_shape) / 2
    image_center -= 0.5

    # + 90 deg for the conventions in HCI
    y_shift = separation_pixel * np.cos(np.deg2rad(pos_angle_deg + 90))
    x_shift = separation_pixel * np.sin(np.deg2rad(pos_angle_deg + 90))

    y_pos = image_center[0] + y_shift
    x_pos = image_center[1] + x_shift

    # create a round mask around the (x_pos, y_pos)
    # position with radius = aperture_radius
    y, x = np.ogrid[
           -x_pos:image_shape[0] - x_pos,
           -y_pos:image_shape[1] - y_pos]

    mask = x ** 2 + y ** 2 <= aperture_radius ** 2

    return mask
