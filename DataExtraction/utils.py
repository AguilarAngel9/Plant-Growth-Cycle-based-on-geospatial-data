# Data Processor Library.
# Authors: THEFFFTKID.

from typing import Dict, List, Tuple, Union
from skimage import exposure, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pathlib
import re


def load_landsat_image(
    img_folder: Union[str, None],
    bands: Union[List[str], None]
) -> Dict:
    """
    Take a folder path and return a dict with the raw vectors extracted from the Earth Engine.
    """
    # Dictionary to save the image.
    images_dict = {}

    if img_folder:
        # Use the provided path.
        path = pathlib.Path(img_folder)
    else:
        # Get the path to retrieve.
        path = pathlib.Path(__file__).parent

    # Get the list of all files.
    files = [f.name for f in path.glob('**/*.tif')]
    # Parse all of filenames to get the unique ones.
    files = set([re.search('_[0-9](.*)[0-9]_', x).group() for x in files])
    # Dict of images to return.
    images_dict = {}

    # Iterate over the files.
    for pat in files:
        image = {}
        # Iterate over the bands.
        for band in bands:
            file = next(path.glob(f'*{pat}{band}.tif'))
            print(f'Opening file {file}')
            ds = rasterio.open(file)
            image.update({band: ds.read(1)})
        # Update the main dict.
        images_dict.update(
            {pat.replace('_','') : image}
        )

    return images_dict


def display_rgb(
    img: Union[Dict, None], 
    alpha=1., 
    figsize=(5, 5)
    ) -> None:
    """
    Display the LANDSAT images as RGB images.
    """
    # Stack the vectors.
    rgb = np.stack(
        [img['B4'], img['B3'], img['B2']],
        axis=-1
    )

    # Reescale.
    rgb = rgb/rgb.max() * alpha
    plt.figure(figsize=figsize)
    plt.imshow(rgb)

# def scale_to_rgb(
#     img: Union[Dict, None],
#     alpha: float = 1.,
#     figsize: Tuple =(5,5)
# ) -> np.array:
#     """
#     Function to scale an imahe to RGB scale [0, 255].
#     """
#     # Stack the vectors.
#     img_stack_ = np.stack(
#         [img['B4'], img['B3'], img['B2']]
#         , axis=-1)
    
#     # Scale the data to [0, 255] to show as an RGB image.
#     # Adapted from https://bit.ly/2XlmQY8. Credits to Justin Braaten.
#     rgb_img_test = (255 * ((img_stack_[:, :, 0:3] - 100) / 3500)).astype('uint8')
    
#     return rgb_img_test

def convert_to_eight_bits(
    img: Union[Dict, None]
) -> np.array:
    """
    To reescale image to 8 bits.
    """
    img_stack = np.stack(
        [img['B4'], img['B3'], img['B2']]
        , axis=-1)

    scaled_img = img_as_ubyte(
        exposure.rescale_intensity(img_stack)
    )
    
    return scaled_img