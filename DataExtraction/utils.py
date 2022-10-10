# Data Processor Library.
# Authors: THEFFFTKID.

from typing import Dict, List, Tuple, Union
from skimage import exposure, img_as_ubyte
from datetime import datetime
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


def stack_to_dict(
    stack: Union[np.stack, None],
    bands: List[str] = ['B4','B3','B2', 'B8']
) -> Dict:
    """
    Unstack the rescaled dictionary.
    """
    unstack_dict = {}

    # Get the dimension of the stack.
    columns = stack.shape[1]

    # Iterate over the columns.
    for i in range(columns):
        band_array = [c[i] for c in stack]
        
        unstack_dict.update(
            {bands[i] : band_array}
        )

    return unstack_dict


def convert_dict_eight_bits(
    images_dict: Union[Dict, None],
) -> Dict:
    """
    Rescale each band to eigth bits.
    """
    reescaled_images = {}

    for key, img in images_dict.items():
        eight_bits_bands = {}
        
        for band in img.keys():
            rescaled_channel = img_as_ubyte(
                exposure.rescale_intensity(img[band])
            )

            eight_bits_bands.update(
                {band : rescaled_channel}
            )
        
        reescaled_images.update(
            {key : eight_bits_bands}
        )

    return reescaled_images


def display_rgb(
    img: Union[Dict, None], 
    title: str ='Landsat',
    alpha=1., 
    figsize=(5, 5)
    ) -> np.array:
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
    plt.title(title)
    plt.imshow(rgb)

    return rgb


def sort_dict_by_date(
    images_dicts : Union[Dict, None]
) -> List:
    """
    Sorted the Images Dict keys' to make a time series analysis.
    """
    # Get all the keys of the dictionary.
    dates_list = [(key, re.findall(f"(\d+)T", key)[0]) for key in images_dicts.keys()]
    # Cast the keys to daterimes.
    datetimes_list = [(date[0], datetime.strptime(date[1], '%Y%m%d')) for date in dates_list]
    # Sort the tuples based on the datetimes.
    datetimes_list = sorted(datetimes_list, key = lambda x: x[1])
    # Extract just the sorted keys.
    keys_sorted_list = [key[0] for key in datetimes_list]

    return keys_sorted_list