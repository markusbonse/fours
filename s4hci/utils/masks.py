import numpy as np
from skimage.morphology import dilation, disk
from scipy.ndimage import shift
from photutils.aperture import CircularAperture


# The new function using percentage of flux
def construct_rfrr_template(flux_coverage,
                            psf_template_in):
    min_flux = np.min(psf_template_in)
    max_flux = np.max(psf_template_in)

    for tmp_threshold in np.linspace(min_flux, max_flux, 1000):
        tmp_template_mask = np.array(psf_template_in >= tmp_threshold,
                                     dtype=int)
        tmp_template = psf_template_in * tmp_template_mask

        tmp_flux_coverage = np.sum(tmp_template) / np.sum(psf_template_in)

        if tmp_flux_coverage < flux_coverage:
            return tmp_template, tmp_template_mask


# The old mask
def construct_circular_rfrr_template(dilatation,
                                     psf_template_in):
    threshold = np.max(psf_template_in)
    template_mask = np.array(psf_template_in >= threshold, dtype=int)

    template_mask = dilation(template_mask, disk(dilatation))
    template = psf_template_in * template_mask

    return template, template_mask


def construct_round_rfrr_template(radius,
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


def construct_rfrr_mask(template_setup,
                        psf_template_in,
                        mask_size_in: int):
    # 1.) Create circular mask around the max PSF template value
    if template_setup[0] == "percent":
        template, template_mask = construct_rfrr_template(
            template_setup[1],
            psf_template_in)

    elif template_setup[0] == "radius":
        template, template_mask = construct_round_rfrr_template(
            template_setup[1],
            psf_template_in)

    elif template_setup[0] == "dilatation":
        template, template_mask = construct_circular_rfrr_template(
            template_setup[1],
            psf_template_in)
    else:
        raise ValueError("mask type not supported")

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


def construct_planet_mask(frame_size,
                          fwhm):

    mask = np.ones((frame_size, frame_size))
    outer_size = frame_size//2
    inner_size = np.ceil(fwhm)

    if frame_size % 2 == 0:
        x_grid = y_grid = np.linspace(-frame_size / 2 + 0.5,
                                      frame_size / 2 - 0.5, frame_size)
    else:
        x_grid = y_grid = np.linspace(-(frame_size - 1) / 2,
                                      (frame_size - 1) / 2, frame_size)

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    rr_grid = np.sqrt(xx_grid ** 2 + yy_grid ** 2)

    mask[rr_grid < inner_size] = 0.
    mask[rr_grid > outer_size] = 0.

    return mask
