import numpy as np


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_validation_positions(
        separation,
        test_image,
        num_positions):

    image_size_radius = int((test_image.shape[0] - 1) / 2)

    angles = np.linspace(0, 2*np.pi, num_positions)
    positions = np.round(np.dstack(
        pol2cart(separation,
                 angles))[0]).astype(int)

    positions += image_size_radius

    return positions
