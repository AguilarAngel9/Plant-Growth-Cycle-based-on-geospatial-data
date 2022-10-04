# Technical notes

## Classic image processing tools. 
- Torres-Sanchez et al. (Torres-Sánchez et al., 2015) utilized object based image analysis on UAV images to detect vegetation in a specific area by calculating two vegetation indices viz. 
- Normalized Difference Vegetation Index (NDVI) and Excess Green (ExG). Wan et al. (Wan et al., 2020) extracted spectral and structural features from RGB images of rice field during its growth period to enhance the grain yield. 
- Li et al. (Li et al., 2018) used half-Gaussian fitting method to estimate the covering of corn crops in a farmland from its UAV image. 
- Enciso et al. (Enciso et al., 2019) used UAV RGB images to assess the height, NDVI and area covered by three different types of tomato plants.
- Weed mapping was done by Stroppiana et al. (Stroppiana et al., 2018) using unsupervised clustering algorithm to detect weed and non-weed areas from UAV RGB images of farmland.
- Support Vector Machine



## Tipos de segmentación
- Clasificación basada en el enfoque (Enfoque basado en regiones (detección de similitudes) y Enfoque basado en límites (detección de discontinuidad))
- Clasificación basada en técnicas(Técnicas Estructurales, estocásticas e híbridas)
las técnicas de segmentación basadas en los valores de pıxel. El segundo grupo contiene técnicas de segmentación basadas en el área. Las técnicas basadas en orillas pertenecen al tercer grupo y finalmente las técnicas basadas en la física es el cuarto grupo.
## Técnicas de segmentación
- SLIC
El algoritmo SLIC (Simple Linear Iterative Clustering) fue introducido por R.Achanta con la finalidad de generar un algoritmo simple y eficiente a comparación de otros algoritmos (Quickshift o N cut). Lo que se realiza es que se debe de indicar un parámetro K, el cual corresponde al número de superpixels (cantidad de pixeles en los que se dividirá la imagen (no todos son del mismo tamaño pero si muy parecidos).  A continuación se presenta un pseudocodigo obtenido de : https://oa.upm.es/69954/1/TFC_BOUDRIIYA_EL_HANDRIS_MOHAMMED_SALIM.pdf con el objetivo de ejemplificar el funcionamiento de este algoritmo.
<p align="center">
  <![image](https://user-images.githubusercontent.com/111094131/191816901-fb7cc23f-c1b4-4a94-a8f5-6121936e7ffe.png)>
</p>




- Segmentación basada en bordes

- Segmentación de umbral

- Segmentación basada en regiones

- Segmentación de cuencas hidrográficas

- Algoritmos de segmentación basados en agrupamiento

- QuickShift

- Redes neuronales para segmentación

Links a revisar
- PyObia: Segmentación y caracterización de imágenes de satélite
https://oa.upm.es/69954/1/TFC_BOUDRIIYA_EL_HANDRIS_MOHAMMED_SALIM.pdf

- Segmentación de imágenes en OpenCV https://www.delftstack.com/es/howto/python/opencv-segmentation/#:~:text=opencv%20en%20Python.-,Segmentaci%C3%B3n%20de%20im%C3%A1genes%20usando%20opencv%20en%20Python,Estas%20curvas%20se%20denominan%20contornos.

- Técnicas de Segmentación en Procesamiento Digital de Imágenes (En detección de bordes se han presentado las técnicas: a) Derivada de primer orden, el operador gradiente, b) Detección de bordes utilizando derivadas de segundo orden, el operador laplaciano, y c) Técnicas de enlazado de bordes y detección de límites.s)
https://sisbib.unmsm.edu.pe/BibVirtual/Publicaciones/risi/2009_n2/v6n2/a02v6n2.pdf

- Segmentacion de imágenes de color
https://www.scielo.org.mx/pdf/cys/v8n4/v8n4a5.pdf

- Image Segmentation Techniques [Step By Step Implementation]
https://www.upgrad.com/blog/image-segmentation-techniques/


# Measuring vegetation  (NDVI)
## Normalized Difference Vegetation Index (NDVI)
This vegetation method is used when there is a need to identify abnormal vegetation and changes in health. This is done because the plant absorbs more visible red light and reflects more NIR than a diseased or old plant whose level of visible red light reflectance is higher and the NIR it reflects is lower. This is because its health is related to the level of chlorophyll in the plant. 
![image](https://user-images.githubusercontent.com/111094131/193896755-e1883078-b1f3-4e07-bf94-2ef90c74585d.png)

The calculation of the NDVI is given by :
$NDVI = (NIR - VISR)/ (NIR + VISR)$
- NIR light reflected in the near-infrared spectrum
- VISR light reflected in the red range of the spectrum

Where the range of NDVI values per pixel should be between (-1) and (+1).
When there is no vegetation the value is 0, non-green leaves give a value close to 0. Values close to +1 (0.8 - 0.9) indicate a large presence of green leaves. It detects vegetation that is of importance to analyse from values of 0.22 onwards.
- Negative values refer to the presence of water bodies ( 0 to -1)
- Values less than or equal to 0.1 can be swept rock, sand and snow. 
- Values between 0.2 to 0.5 can be shrubs, grassland and senescent (old) crops.
- Values between 0.6 to 0.9 dense vegetation present

Source: [NASA Earth Observatory](https://earthobservatory.nasa.gov/features/MeasuringVegetation/measuring_vegetation_2.php)
## Soil Adjusted Vegetation Index (SAVI)
It is used to correct NDVI for the influence of shiny oil in areas with little vegetation, i.e. it is an alternative method to NDVI specialised in young crops and arid areas. The results are equally between (-1) and (1).

The calculation of the SAVI is given by :
$SAVI = (NIR - VISR) / (NIR + VISR + L)  * (1 + L)$
- NIR light reflected in the near-infrared spectrum
- VISR light reflected in the red range of the spectrum
- L correction factor for soil brightness

The L value is given by the conditions;
- L=1 in areas with moderate vegatative
- L=0.5 in areas with very high vegetation
- L=0 is equal to NDVI method
The value of L is adjusted based on the amount of vegetation. L=0.5 is the default value and works well in most situations

Source: [USGS](https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index)

## Green Chlorophyll Index (GCI)
This method is used to estimate leaf chlorophyll content in the plants based only on NIR and VISG. THe value of chlorophyll directly reflects the vegetation's existence.

The calculation of the GCI is given by :
$CGI = (NIR)/ (VISG)  - 1 $

- NIR light reflected in the near-infrared spectrum
- VISG light reflected in the green range of the spectrum


Source: [EOS](https://eos.com/make-an-analysis/chlorophyll-index/)


## Supervised learning tecniques
- Malek et al. (Malek et al., 2014) proposed automatic palm tree detection algorithm to identify haphazardly planted palm oil trees from UAV RGB images of the land. In the proposed identification process, feature extraction was done using Scale-Invariant Feature Transformation (SIFT) and classification was performed by extreme learning machine classifier.
- Chew et al. (Chew et al., 2020) identified three different food crops (banana, maize and legume) using transfer learning process from pre trained VGG-16 and ImageNet CNN modules.
- However, the previous literatures mainly performed binary classification with the UAV RGB images i.e. segregation of crop region (viz. rice, weed and tobacco, or tree, planted area) from non-crop region.

## Code propose
The regular approaches have used convnets for semantic segmentation, in which each pixel is labeled withthe class of its enclosing object or region, but with short-comings that this work addresses.
