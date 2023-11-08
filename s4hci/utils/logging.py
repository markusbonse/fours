import logging
import sys
import numpy as np
from copy import deepcopy


def setup_logger() -> logging.Logger:
    """
    Create a new logger object.
    """

    # Create a new logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set up a new formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up a Stream handler to write logs to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the SimpleLogger object
    logger.addHandler(stream_handler)

    return logger


def print_message(message_in):
    logger = logging.getLogger("main")
    logger.info(message_in)
    del logger


def normalize_for_tensorboard(frame_in):
    image_for_tb = deepcopy(frame_in)
    image_for_tb -= np.min(image_for_tb)
    image_for_tb /= np.max(image_for_tb)
    return image_for_tb
