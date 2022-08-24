# Technical notes

## Classic image processing tools. 
- Torres-Sanchez et al. (Torres-Sánchez et al., 2015) utilized object based image analysis on UAV images to detect vegetation in a specific area by calculating two vegetation indices viz. 
- Normalized Difference Vegetation Index (NDVI) and Excess Green (ExG). Wan et al. (Wan et al., 2020) extracted spectral and structural features from RGB images of rice field during its growth period to enhance the grain yield. 
- Li et al. (Li et al., 2018) used half-Gaussian fitting method to estimate the covering of corn crops in a farmland from its UAV image. 
- Enciso et al. (Enciso et al., 2019) used UAV RGB images to assess the height, NDVI and area covered by three different types of tomato plants.
- Weed mapping was done by Stroppiana et al. (Stroppiana et al., 2018) using unsupervised clustering algorithm to detect weed and non-weed areas from UAV RGB images of farmland.
- Support Vector Machine


## Supervised learning tecniques
- Malek et al. (Malek et al., 2014) proposed automatic palm tree detection algorithm to identify haphazardly planted palm oil trees from UAV RGB images of the land. In the proposed identification process, feature extraction was done using Scale-Invariant Feature Transformation (SIFT) and classification was performed by extreme learning machine classifier.
- Chew et al. (Chew et al., 2020) identified three different food crops (banana, maize and legume) using transfer learning process from pre trained VGG-16 and ImageNet CNN modules.
- However, the previous literatures mainly performed binary classification with the UAV RGB images i.e. segregation of crop region (viz. rice, weed and tobacco, or tree, planted area) from non-crop region.

## Code propose
The regular approaches have used convnets for semantic segmentation, in which each pixel is labeled withthe class of its enclosing object or region, but with short-comings that this work addresses.