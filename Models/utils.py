# Data Processor Library.
# Authors: THEFFTKID.

from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Union
from skimage import exposure, img_as_ubyte
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from operator import itemgetter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio
import pathlib
import cv2
import re

#  Data collection and preprocession 
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
            #print(f'Opening file {file}')
            ds = rasterio.open(file)
            image.update({band: ds.read(1)})
        # Update the main dict.
        images_dict.update(
            {pat.replace('_','') : image}
        )

    return images_dict

def convert_to_eight_bits(
    img: Union[Dict, None]
) -> np.ndarray:
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
    alpha: float = 1.0, 
    figsize: Tuple =(5,5)
    ) -> np.ndarray:
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
    
    # Create.
    unstack_dict = {bands[i] : bands_lst[i]  for i in range(len(bands))}
            
    return unstack_dict

# Get center pixels
def get_center_pixels(
    image_data: Dict, 
    square_shape: Tuple = (10,10)
) -> np.ndarray:
    """
    Takes the center area of a picture of shape square_shape
    """
    drawing = image_data.copy()
    h, w, _ = image_data.shape
    rows, cols = square_shape
    
    center = (round(h/2), round(w/2))

    square = image_data[
        center[0] - cols : center[0] + cols,
        center[1] - rows : center[1] + rows
    ]

    cv2.rectangle(
        drawing
        , (center[0]-cols, center[1]-rows)
        , (center[0]+cols, center[1]+rows)
        , (255, 255, 255)
    )

    plt.imshow(drawing)

    return square

# Indices calculation and plots
def calculate_ndvi(
    image_data: Union[Dict, None],
    square_shape: Tuple = (10,10),
    visualize: bool = False
) -> np.ndarray:
    """
    Implementation of the Normalized Difference Vegetation Index (NDVI).
    """
    # Get the shape of the interest zone.
    h, l = square_shape
    # Reshape the arrays.
    b4 = np.array(image_data["B4"]).reshape(2*h, 2*l, 1)
    b8 = np.array(image_data["B8"]).reshape(2*h, 2*l, 1)

    # Cast the values to float.
    visr = b4.astype("float64")
    nir = b8.astype("float64")

    # Calculate the NDVI over the matrix.
    ndvi_matrix = np.where((nir+visr)==0.0, 0, (nir-visr)/(nir+visr))

    if visualize:
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(ndvi_matrix, cmap='viridis')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return ndvi_matrix

def calculate_wdrvi(
    image_data: Union[Dict, None],
    square_shape: Tuple = (10,10),
    a: float = 0.1,
    visualize: bool = False
) -> np.ndarray:
    """
    Implemenation of the Wide Dynamic Range Vegetation Index (WDRI).
    """
    # Get the shape of the interest zone.
    h, l = square_shape
    # Reshape the arrays.
    b4 = np.array(image_data["B4"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)
    
    # Cast the values to float.
    visr = b4.astype("float64")
    nir = b8.astype("float64")
    
    # Calculate the WDRVI over the matrix
    wdrvi_matrix = np.where((nir+visr)==0., 0, (a*nir-visr)/(a*nir+visr))
    
    if visualize:
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(wdrvi_matrix, cmap='viridis')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return wdrvi_matrix

def calculate_savi(
    image_data: Union[Dict, None],
    square_shape: Tuple = (10,10),
    L: float = 0.5,
    visualize = False
) -> np.ndarray:
    """
    Implemenation of the Soil Adjusted Vegetation Index (SAVI).
    """
    # Get the shape of the interest zone.
    h, l = square_shape
    # Reshape the arrays.
    b4 = np.array(image_data["B4"]).reshape(2*h, 2*l, 1)
    b8 = np.array(image_data["B8"]).reshape(2*h, 2*l, 1)
    
    # Cast the values to float.
    visr = b4.astype("float64")
    nir = b8.astype("float64")

    # Calculate the SAVI over the matrix
    savi_matrix=np.where((visr+nir + L)==0., 0, ((nir-visr)/(visr+nir + L) ) * (1+L))

    if visualize:
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    
        im = ax.imshow(savi_matrix, cmap='RdYlGn')

        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return savi_matrix

def calculate_gci(
    image_data: Dict,
    square_shape: Tuple = (10,10),
    visualize: bool = False
) -> np.ndarray:
    """
    Implemenation of the Wide Green Chlorophyll Index (GCI).
    """
    # Get the shape of the interest zone.
    h, l = square_shape
    # Reshape the arrays.
    b3 = np.array(image_data["B3"]).reshape(2*h,2*l,1)
    b8 = np.array(image_data["B8"]).reshape(2*h,2*l,1)
    
    # Cast the values to float.
    visg = b3.astype('float64')
    nir = b8.astype('float64')
    
    # Calculate the GCI over the matrix
    gci_matrix = np.where((visg)==0.0, 0, (nir)/(visg) - 1)

    if visualize:
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(gci_matrix, cmap='RdYlGn')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

    return gci_matrix

def calculate_index_avg(
    index_matrix: np.ndarray
) -> float:
    """
    Get the average of the matrix calculated index.
    """
    index_avg = index_matrix.flatten().sum()/len(index_matrix.flatten())
    return index_avg

def generate_ndvi_time_series(
    images: Union[Dict, None]
) -> List:
    """
    Calculate the NDVI time series for a given dict of images.
    """
    # NDVI time series.
    ndvi_values = []

    # Iterate over the dictionary.
    for image in sort_dict_by_date(images):
        # Transform the current image to 8-bits.
        img = convert_to_eight_bits(images[image])
        # Get the center (l x l) of the current image.
        representative_region = get_center_pixels(img)
        # Unstack the current np.stack to dict.
        representative_region_bands = stack_to_dict(representative_region)
        # Calculate the NDVI over the region.
        index_calculation = calculate_ndvi(representative_region_bands)
        # Get the average value over the selected area.
        average = calculate_index_avg(index_calculation)

        ndvi_values.append(average)

    return ndvi_values

def generate_savi_time_series(
    images: Union[Dict, None]
) -> List:
    """
    Calculate the SAVI time series for a given dict of images.
    """
    # SAVI time series.
    savi_values = []
    
    # Iterate over the dictionary.
    for image in sort_dict_by_date(images):
        # Transform the current image to 8-bits.
        img = convert_to_eight_bits(images[image])
        # Get the center (l x l) of the current image.
        representative_region = get_center_pixels(img)
        # Unstack the current np.stack to dict.
        representative_region_bands = stack_to_dict(representative_region)
        # Calculate the SAVI over the region.
        index_calculation = calculate_savi(representative_region_bands)
        # Get the average value over the selected area.
        average = calculate_index_avg(index_calculation)

        savi_values.append(average)

    return savi_values

def generate_gci_time_series(
    images: Union[Dict, None]
) -> List:
    """
    Calculate the GCI time series for a given dict of images.
    """
    # GCI time series.
    gci_values = []

    # Iterate over the dictionary.
    for image in sort_dict_by_date(images):
        # Transform the current image to 8-bits.
        img = convert_to_eight_bits(images[image])
        # Get the center (l x l) of the current image.
        representative_region = get_center_pixels(img)
        # Unstack the current np.stack to dict.
        representative_region_bands = stack_to_dict(representative_region)
        # Calculate the GCI over the region.
        index_calculation = calculate_gci(representative_region_bands)
        # Get the average value over the selected area.
        average = calculate_index_avg(index_calculation)

        gci_values.append(average)
    
    # Perform a normalization
    norm = [float(i)/max(gci_values) for i in gci_values] 
    return norm

def generate_wdrvi_time_series(
    images: Union[Dict, None]
) -> List:
    """
    Calculate the WDRVI time series for a given dict of images.
    """
    # WDRVI time series.
    wdrvi_values = []

    # Iterate over the dictionary.
    for image in sort_dict_by_date(images):
        # Transform the current image to 8-bits.
        img = convert_to_eight_bits(images[image])
        # Get the center (l x l) of the current image.
        representative_region = get_center_pixels(img)
        # Unstack the current np.stack to dict.
        representative_region_bands = stack_to_dict(representative_region)
        # Calculate the GCI over the region.
        index_calculation = calculate_wdrvi(representative_region_bands)
        # Get the average value over the selected area.
        average = calculate_index_avg(index_calculation)

        wdrvi_values.append(average)
    
    # Perform a normalization.
    norm = [float(i)/max(wdrvi_values) for i in wdrvi_values] 
    return norm

def dates_to_day_numbers(
    img_keys: List[str]
) -> Tuple[List, List]:
    """
    Changes the images dates to the natural number day.
    """
    # List of dates.
    dates_list = []

    # Iterate over the key list.
    for image_details in img_keys:
        # Parse the date from the key.
        date = pd.to_datetime(image_details[0:8])
        # Format Y-m-d
        day_format = date.strftime('%Y-%m-%d')
        dates_list.append(day_format)
    
    dates_list.sort()

    # List of numbers.
    initial_date = datetime.strptime(dates_list[0], '%Y-%m-%d')
    # Calculate the differences between the initial and the nexts days.
    day_numbers = [datetime.strptime(day, '%Y-%m-%d') - initial_date for day in dates_list]
    # Get the difference in days.
    day_numbers = [day // timedelta(days=1) for day in day_numbers]

    return day_numbers, dates_list

# Curve smoothing
def match_indexes(
    indexes: Union[List, None],
    array_to_match: Union[List, None]
) -> List:
    """
    To match an index list to it's reference.
    """
    try:
        intersection = itemgetter(*indexes)(array_to_match)
    except TypeError:
        intersection = []

    # Cast to list.
    if type(intersection) == tuple:
        intersection = list(intersection)
    elif type(intersection) == np.int64:
        intersection = [intersection]

    return intersection

def identify_outliers(
    raw_x: Union[List, np.ndarray],
    raw_y: Union[List, np.ndarray],
    outliers_fraction : float = 0.10
) -> List:
    """
    Identify an Isolation Forest for outiler identification.
    """
    if raw_x != np.ndarray and raw_y != np.ndarray:
        # Cast and reshape.
        reshaped_y = np.array(raw_y).reshape(-1, 1)
    else:
        reshaped_y = raw_y.reshape(-1, 1)
    
    # Instance the Isolation Forest.
    outliers_model = IsolationForest(
        contamination=outliers_fraction
    )

    # Fit & Predict the raw data.
    new_data = outliers_model.fit_predict(
        reshaped_y
    )

    # Get the indexes of the outliers.
    outliers_ind = [index for index, value in enumerate(new_data) if value == -1]
    outliers = match_indexes(outliers_ind, raw_y)

    # Get the indexes of the good values.
    clean_ind = [index for index, value in enumerate(new_data) if value == 1]
    clean_y = match_indexes(clean_ind, raw_y)
    # Get the x values.
    clean_x = match_indexes(clean_ind, raw_x)

    return clean_x, clean_y

def preprocess_data(
    raw_x : Union[List, np.ndarray],
    raw_y : Union[List, np.ndarray],
    smoothing_filter : str = 'lowess'
) -> Tuple:
    """
    Preprocess the raw data, applying an Unsupervised Outlier Detection and an Smoothing Filter.
    """
    # Get the outliers.
    transformed_x, transformed_y = identify_outliers(
        raw_x=raw_x , raw_y=raw_y
    )
    
    # Smooth the curve.
    if smoothing_filter == 'savitzky':
        # Savitzky-Golay polynomial smoothering.
        smoothered_y = savgol_filter(transformed_x, window_length=7, polyorder=3)
    elif smoothing_filter == 'lowess':
        # Locally Weighted Scatterplot Smoothing
        smoothered_y =  lowess(
            transformed_y, transformed_x, is_sorted=False, frac=0.15, it=0, return_sorted=False
        )

    return transformed_x, smoothered_y

