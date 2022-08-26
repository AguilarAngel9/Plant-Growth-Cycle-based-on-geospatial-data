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