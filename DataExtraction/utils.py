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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import pandas as pd
import cv2


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
        [img['B4'], img['B3'], img['B2'], img['B8']]
        , axis=-1)

    scaled_img = img_as_ubyte(
        exposure.rescale_intensity(img_stack)
    )
    
    return scaled_img


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


def stack_to_dict(
    stack: Union[np.stack, None],
    bands: List[str] = ['B4','B3','B2','B8']
) -> Dict:
    """
    Unstack the rescaled dictionary.
    """
    dimension = stack.shape

    # Create the arrays of the bands.
    bands_lst = [[] for band in bands]
    
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            # R
            bands_lst[0].append(
                stack[i][j][0]
            )
            # G
            bands_lst[1].append(
                stack[i][j][1]
            )
            # B
            bands_lst[2].append(
                stack[i][j][2]
            )
            # NIR
            bands_lst[3].append(
                stack[i][j][3]
            )
    
    # Create
    unstack_dict = {bands[i] : bands_lst[i]  for i in range(len(bands))}
            
    return unstack_dict

def get_center_pixels(image_data: dict, 
    square_shape=(3,3)) -> np.array:
    """
    Takes the center area of a picture of shape square_shape
    """
    drawing = image_data.copy()
    h, w, _ = image_data.shape
    rows, cols = square_shape
    
    center = (round(h/2), round(w/2))

    square = image_data[center[0]-cols:center[0]+cols, center[1]-rows:center[1]+rows]

    cv2.rectangle(drawing, (center[0]-cols, center[1]-rows), (center[0]+cols, center[1]+rows), (255, 255, 255))
    plt.imshow(drawing)

    return square

def calculate_ndvi(image_data,
    square_shape=(3,3),
    visualize=True) -> np.array:
    
    h, l = square_shape
    b4 = np.array(image_data["B4"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)

    visr = b4.astype("float64")
    nir = b8.astype("float64")

    ndvi_matrix=np.where((nir+visr)==0., 0, (nir-visr)/(nir+visr))

    if visualize == True:
    
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(ndvi_matrix, cmap='viridis')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return ndvi_matrix

def calculate_wdrvi(image_data: dict,
    square_shape=(3,3),
    a=0.1,
    visualize=True) -> np.array:

    h, l = square_shape
    b4 = np.array(image_data["B4"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)

    visr = b4.astype("float64")
    nir = b8.astype("float64")

    wdrvi_matrix=np.where((nir+visr)==0., 0, (a*nir-visr)/(a*nir+visr))
    
    if visualize == True:

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(wdrvi_matrix, cmap='viridis')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return wdrvi_matrix

def calculate_savi(image_data: dict,
    square_shape=(3,3),
    L=0.5,
    visualize=True) -> np.array:

    h, l = square_shape
    b4 = np.array(image_data["B4"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)

    visr = b4.astype("float64")
    nir = b8.astype("float64")

    savi_matrix=np.where((visr+nir + L)==0., 0, ((nir-visr)/(visr+nir + L) ) * (1+L))

    if visualize == True:

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    
        im = ax.imshow(savi_matrix, cmap='RdYlGn')

        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return savi_matrix

def calculate_gci(image_data: dict,
    square_shape=(3,3),
    visualize=True) -> np.array:

    h, l = square_shape
    b3 = np.array(image_data["B3"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)

    visg = b3.astype('float64')
    nir = b8.astype('float64')

    gci_matrix=np.where((visg)==0., 0, (nir)/(visg) - 1)

    if visualize == True:

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(gci_matrix, cmap='RdYlGn')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return gci_matrix

def calculate_index_avg(index_matrix: np.array) -> int:
    return index_matrix.flatten().sum()/len(index_matrix.flatten())

def generate_ndvi_time_series(images: dict) -> list:

    values = []

    for image in sort_dict_by_date(images):
        img = convert_to_eight_bits(images[image])
        representative_region = get_center_pixels(img)
        representative_region_bands = stack_to_dict(representative_region)
        index_calculation = calculate_ndvi(representative_region_bands)
        average = calculate_index_avg(index_calculation)

        values.append(average)

    return values

def generate_savi_time_series(images: dict) -> list:

    values = []

    for image in sort_dict_by_date(images):
        img = convert_to_eight_bits(images[image])
        representative_region = get_center_pixels(img)
        representative_region_bands = stack_to_dict(representative_region)
        index_calculation = calculate_savi(representative_region_bands)
        average = calculate_index_avg(index_calculation)

        values.append(average)

    return values

def generate_gci_time_series(images: dict) -> list:


    values = []

    for image in sort_dict_by_date(images):
        img = convert_to_eight_bits(images[image])
        representative_region = get_center_pixels(img)
        representative_region_bands = stack_to_dict(representative_region)
        index_calculation = calculate_gci(representative_region_bands)
        average = calculate_index_avg(index_calculation)

        values.append(average)
    #normalization
    norm = [float(i)/max(values) for i in values] 
    return norm

def generate_wdrvi_time_series(images: dict) -> list:
    

    values = []

    for image in sort_dict_by_date(images):
        img = convert_to_eight_bits(images[image])
        representative_region = get_center_pixels(img)
        representative_region_bands = stack_to_dict(representative_region)
        index_calculation = calculate_wdrvi(representative_region_bands)
        average = calculate_index_avg(index_calculation)

        values.append(average)
    #normalization
    norm = [float(i)/max(values) for i in values] 
    return norm


