# Data Processor Library.
# Authors: THEFFTKID.

from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, date
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Union
from skimage import exposure, img_as_ubyte
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from operator import itemgetter
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import rasterio
import cdsapi
import pathlib
import cv2
import re
from calendar import monthrange

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
            # print(f'Opening file {file}')
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
    images_dict: Union[Dict, None]
) -> Dict:
    """
    Rescale each band to eigth bits.
    """
    reescaled_images = {}
    # Iterate over the images.
    for key, img in images_dict.items():
        eight_bits_bands = {}
        # Iterate over the bands.
        for band in img.keys():
            rescaled_channel = img_as_ubyte(
                exposure.rescale_intensity(img[band])
            )
            # Add a reescaled band.
            eight_bits_bands.update(
                {band : rescaled_channel}
            )
        # Update main dict.
        reescaled_images.update(
            {key : eight_bits_bands}
        )

    return reescaled_images

def display_rgb(
    img: Union[Dict, None], 
    title: str ='Landsat',
    alpha: float = 1.0, 
    figsize: Tuple =(5, 5)
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

def get_center_pixels(
    image_data: Dict, 
    square_shape: Tuple = (3, 3)
) -> np.ndarray:
    """
    Takes the center area of a squared-shape picture.
    """
    # Deep copy the data.
    drawing = image_data.copy()
    # Get the shape of the image.
    h, w, _ = image_data.shape
    # Shape to crop.
    rows, cols = square_shape
    
    center = (round(h/2), round(w/2))
    # Select the zone.
    square = image_data[
        center[0] - cols : center[0] + cols,
        center[1] - rows : center[1] + rows
    ]
    # Draw the trimmed area.
    cv2.rectangle(
        drawing
        , (center[0]-cols, center[1]-rows)
        , (center[0]+cols, center[1]+rows)
        , (255, 255, 255)
    )

    plt.imshow(drawing)

    return square

def calculate_ndvi(
    image_data: Union[Dict, None],
    square_shape: Tuple = (3, 3),
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
    ndvi_matrix = np.where(
        (nir + visr) == 0.0, 0, (nir - visr)/(nir + visr)
    )

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
    square_shape: Tuple = (3,3),
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
    wdrvi_matrix = np.where(
        (nir+visr) == 0., 0, (a * nir - visr)/(a * nir + visr)
    )
    
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
    square_shape: Tuple = (3,3),
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
    savi_matrix=np.where(
        (visr + nir + L) == 0., 0, ((nir - visr)/(visr + nir + L) ) * (1 + L)
    )

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
    square_shape: Tuple = (3, 3),
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
    gci_matrix = np.where((visg) == 0.0, 0, (nir)/(visg) - 1)

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

def images_time_info(
    img_keys: List[str],
    initial_date: date
) -> Tuple [List, List, List]:
    """
    Changes the images dates to the natural number day after query begins.
    Returns list of natural number days, list of dates, list of hours.
    Initial date must be in datetime.date(Y,m,d) format.
    """
    # Dates, hours and timestamps.
    dates_list = []
    hours_list = []
    timestamps_list = []

    # Iterate over the key list.
    for image_details in img_keys:
        # Parse the date from the key.
        date = pd.to_datetime(image_details[0:15])
        day_format = date.strftime('%Y-%m-%d')
        dates_list.append(day_format)
        timestamps_list.append(date)
        # Hours from images retrieved
        hour_of_day= date.round(freq='H')
        hour_of_day = hour_of_day.strftime('%H:%M')
        hours_list.append((hour_of_day))

    # Sorts.
    hours_list = set(hours_list)
    dates_list.sort()

    # List of numbers.
    initial_date = datetime(timestamps_list[0].year, initial_date.month, initial_date.day)
    # Calculate the differences between the initial and the nexts days.
    day_numbers = [datetime.strptime(day, '%Y-%m-%d') - initial_date for day in dates_list]
    # Get the difference in days.
    day_numbers = [day // timedelta(days=1) for day in day_numbers]

    return day_numbers, sorted(timestamps_list), hours_list

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

    # Fit and predict the raw data.
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
    raw_x: Union[List, np.ndarray],
    raw_y: Union[List, np.ndarray],
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

def interpolate_curve(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    n_points: int = 100
) -> Tuple[List]:
    """
    Function to interpolate the curve of the given a N expected points.
    A 1D array are assumed and the boundary conditions of the second derivative
    at curve ends are zero.  
    """
    f_x = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(min(x), max(x), n_points)
    y_new = f_x(x_new)

    return x_new, y_new

def data_extrator_temp(
    data_tp: xr.DataArray ,
    date_dict_api: Dict
) -> Dict:
    # Create de dict for the values and their dates
    data_dict = {}
    # Obtain the temperature values.
    temperature = data_tp['t2m'].values.ravel()
    # Obtain the precipitation values.
    precipitation = data_tp['tp'].values.ravel()

    # Years to the data.
    years = date_dict_api['year']
    years = [int(year) for year in years]
    # Ordered month.
    months = list(set([x.to_pydatetime().month for x  in data_tp['time'].to_series()]))

    # Iteration to aggregate the corresponding values per month.
    # Day and temperature values are added.
    for year in years:
        year_temp = {}
        for month in sorted(months):
            # Number of days in a month,
            month_range = monthrange(year, month)[1]
            # Generate days in the month.
            days = [x + 1 for x in range(month_range)]
            # Number of temperature and precipitation data per day.
            n_data= len(set([x.to_pydatetime().hour for x  in data_tp['time'].to_series()]))

            month_temp = {}

            # Get the temperature.
            for day in days:
                # Temperature and precipitation per hour.
                values_per_hour = {
                    'temperature' : temperature[0:n_data], 
                    'precipitation' : precipitation[0:n_data]
                }
                # Take the corresponding values per day.
                temperature = temperature[n_data:]
                precipitation = precipitation[n_data:]
                values_t_p = {
                    'temperature' : np.mean(values_per_hour['temperature']) ,
                    'precipitation' : np.mean(values_per_hour['precipitation'])
                }
                # Add the previously calculated values.
                month_temp.update(
                    {day : values_t_p}
                )
            # Months, days, temperature, precipitation data.
            year_temp.update(
                {month : month_temp}
            )
        data_dict.update(
            {year : year_temp}
        )
    return data_dict

def get_temp_and_preci(
    dict_from_temp_extractor: Dict,
    timestamps_list: List
) -> Tuple[List[float], List[float]]:
    """
    Retrieves values of temperature and precipitation from dictionary obtained by the API
    according to the timestamps of our original images.
    """
    # Lists of precipitation and temperature.    
    preci = []
    temp = []
    
    # Iterate over the timestamps list.
    for timestamp in timestamps_list:
        try:
            # Keys obtained to search in dict.
            year, month, day = timestamp.year, timestamp.month, timestamp.day
            
            # Searches for data in dict.
            t_data = float(dict_from_temp_extractor[year][month][day]['temperature'])
            p_data = float(dict_from_temp_extractor[year][month][day]['precipitation'])
            
            temp.append(t_data)
            preci.append(p_data)
        except:
            break
    return temp, preci

def date_range(
    date_min: date, 
    date_max: date
):
    """
    Function that returns a list of the days between two dates,
    inicial date and last date.
    """
    for difference_between_dates in range(int((date_max - date_min).days) + 1):
        # Dates starting at date min and ending at date max.
        yield date_min + timedelta(difference_between_dates)

def temperature_precipitation_api(
    date_min: date, 
    date_max: date,
    hours: List
) -> Tuple[xr.Dataset, Dict]:
    """
    Function that fetches temperature and precipitation data from the API.
    """
    # Create the dictionary to store the dates.
    date_dic = {}
    year_list = []
    month_list = []
    day_list = []
    hour_list = []

    # Iterate in the range of the created dates.
    for dt in date_range(date_min, date_max):
        # Years. 
        year_list.append((dt.strftime("%Y")))
        # Months. 
        month_list.append((dt.strftime("%m")))
        # Days. 
        day_list.append((dt.strftime("%d")))
        # Hours.
        hour_list.append((dt.strftime("%H:%M")))

    # Add the total dates without repetition.
    date_dic.update({
        'year': sorted(set(year_list)),
        'month': sorted(set(month_list)),
        'day': sorted(set(day_list)),
        'hour': sorted(list(hours))
    })
    
    # Call the API to collect data.
    # MUST HAVE THE .cdsapirc FILE.
    c = cdsapi.Client()

    #The api is sent to be called with the values we want.
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature', 'total_precipitation',],
            'year': date_dic['year'],
            'month': date_dic['month'],
            'day': date_dic['day'],
            'time': date_dic['hour'],
            'area': [38.1445082027146, -97.72654627101196, 38.142173460759004, 97.72096882266754,],
            'format': 'netcdf',
        },
        'dataTemperatureCurrent.nc'
    )
    # API payload. 
    data_tp = xr.open_dataset('dataTemperatureCurrent.nc')
    return data_tp, date_dic

def values_temp_precip(
    dict_data: Dict
) -> Tuple[List, List]:
    # Temperature. 
    data_precipitation = np.array(dict_data['tp'])
    # Precipitation.
    data_temperature = np.array(dict_data['t2m'])
    return data_precipitation, data_temperature

def future_temperature_values(
    lat: float,
    lon: float
)-> Tuple[np.ndarray, np.ndarray]:
    # Load the dataset for future temperature.
    data_temperature_future = xr.open_dataset('dataTemperatureFuture.nc')
    
    # Lat and lon values.
    lat_global=np.array(data_temperature_future.lat)
    lon_global=np.array(data_temperature_future.lon)
    
    # Location similar to the given (lat, lon).
    dif_lat = (np.absolute(lat_global-lat))
    dif_lon = (np.absolute(lon_global-lon))
    location_lat = dif_lat.argmin()
    location_lon = dif_lon.argmin()
    
    # Temperature for the retrieve (lat,lon)
    temperature_future = data_temperature_future.tas.isel(lat=location_lat,lon=location_lon)
    
    # Separete the data for years.
    temperature_aprox_2022 = np.array(temperature_future[0:365])
    temperature_aprox_2023 = np.array(temperature_future[365:])
    
    return temperature_aprox_2022, temperature_aprox_2023