from PIL import Image
from pathlib import Path
import numpy as np


def load_image(image_path: Path, image_type: str = 'L') -> np.ndarray:
    """
    Loads the image specified by the input image_path.
    :param image_path: path to the image
    :param image_type: image type.
    :return: numpy ndarray containing the loaded image.
    """
    return np.array(Image.open(image_path).convert(image_type))
