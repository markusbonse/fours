import logging
import sys
import numpy as np
from copy import deepcopy


def setup_logger() -> logging.Logger:
    """
    Sets up and configures a logger object with a specific format and stream 
    handler for output to the console.

    Returns:
        logging.Logger: Configured logger instance with a DEBUG logging level.
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


def print_message(
        message_in: str
) -> None:
    """
    Logs a message at the INFO level using the logger named 'main'.

    Args:
        message_in (str): The string message to be logged.
    """
    
    logger = logging.getLogger("main")
    logger.info(message_in)
    del logger


def normalize_for_tensorboard(
        frame_in: np.ndarray
) -> np.ndarray:
    """
    Normalizes an input image array to be scaled between 0 and 1 for visualization 
    in TensorBoard.

    The function takes a NumPy array as input, creates a deep copy to avoid 
    modifying the original array, and scales it so that the minimum value becomes 
    0 and the maximum value becomes 1.

    Args:
        frame_in (np.ndarray): The input image or array that needs normalization.

    Returns:
        np.ndarray: A normalized version of the input array with values between 
        0 and 1.
    """
    
    image_for_tb = deepcopy(frame_in)
    image_for_tb -= np.min(image_for_tb)
    image_for_tb /= np.max(image_for_tb)
    return image_for_tb
