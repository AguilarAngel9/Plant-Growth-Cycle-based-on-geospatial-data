# Options to retrieve data

One of the main problems facing any machine learning project or product is data availability. Most of the literature reviewed focuses on work with a large data set, around thousands of images, that have been acquired in situ. For this reason, the data acquisition process for this project is one of the most important parts. In this section we will explore some options.

## USGS EarthExplorer
Query and order satellite imagery, aerial photographs and cartographic products through the U.S. Geological Survey. Some of its main advantages are: its graphical interface, its large dataset of Landsat images (more than 20,000) and its free nature. However, it has cons such as manual search and download and low quality image resolution.

[USGS EarthExplorer](https://earthexplorer.usgs.gov/)

## Google Earth Engine
Google Earth Engine is a geospatial processing service, powered by Google Cloud Platform. Earth Engine provides an interactive platform for geospatial algorithm development at scale that involve large geospatial datasets. Despite being build-in in JavaScript, it has an opne source Python library (ee) running on Colab, local Python environment, or App Engine.

So far, we have identified some data sets that could be useful.

- Datasets tagged highres in Earth Engine
    - [NAIP: National Agriculture Imagery Program](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ#description)
    - [Planet SkySat Public Ortho Imagery, RGB](https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_RGB)
    - [Planet SkySat Public Ortho Imagery, Multispectra](https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_MULTISPECTRAL)

- Landsat 9: Landsat, a joint program of the USGS and NASA, has been observing the Earth continuously from 1972 through the present day. Today the Landsat satellites image the entire Earth's surface at a 30-meter resolution about once every two weeks, including multispectral and thermal data. The USGS produces data in 3 categories for each satellite (Tier 1, Tier 2 and RT)
    - [USGS Landsat 9 Level 2, Collection 2, Tier 1](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2)
    - [USGS Landsat 9 Collection 2 Tier 1 TOA Reflectance](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_TOA)
    - [USGS Landsat 9 Collection 2 Tier 1 Raw Scenes](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1)

## Sentinel-2
Sentinel-2 (S2) is a wide-swath, high-resolution, multispectral imaging mission with a global 5-day revisit frequency. The S2 Multispectral Instrument (MSI) samples 13 spectral bands: visible and NIR at 10 meters, red edge and SWIR at 20 meters, and atmospheric bands at 60 meters spatial resolution. It provides data suitable for assessing state and change of vegetation, soil, and water cover.

![Crop times in Germany](https://ipad.fas.usda.gov/rssiws/al/crop_calendar/images/europe_gm_calendar.png)

# Obtaining the testing data
In order to find appropiate geometrical cropland to study we did manual research of corn fields around the globe using the Google Maps api and different websites.
We arbitrarily opt to search the biggest corn producer states in the U.S. and Canada. We finally used crops from Iowa, U.S. and Manitoba, CAN. This for the purpose of finding large and traceable parcels considering the medium resolution of Sentinel-2. The main resources used to find the crops were the following databases:
    - [USGS Global Groplands](https://www.usgs.gov/apps/croplands/app/map?lat=42.50324&lng=-88.68404388427734&zoom=12)
    - [USDA Voyager - Navigo](https://voyager.fas.usda.gov/voyager/navigo/search?sort=score%20desc&disp=D176678659AD&basemap=ESRI%20World%20Street%20Map&view=summary&sortdir=desc&place.op=within&q=tomato%20calendar&filter=true)
    
The USGS Croplands Map is a global map where specific crop types and other geological data are displayed by map layers. In this project it was used to scout for cropfields of our interest by map layering, specifically Global GCE 1km Cropland Dominance and Other Products. Differenced by different colors we mainly focused on the color that included corn as a dominant crop (dark green) and another one that also had the mentioned grain (light green). After finding the locations of some selected cropfields we used the Voyager - Navigo website to get the crops' calendar of the countries. Both websites are owned and managed by the U.S. government (USGS and USDA respectively).
The results are listed in this [spreadsheet]( https://docs.google.com/spreadsheets/d/1w2lCI72LmMnyKiFCDlUVVUWBtV1YHIwRoVhVG5_bnZw/edit#gid=0).

