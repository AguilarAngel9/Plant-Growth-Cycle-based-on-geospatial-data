# Plant Cycle Identification with Satellite Data

Given a farm location, identify the plant growth cycle over the years based on historical data.


## Key Questions
- What kind of information can provide the most insight?
- How can we use these features to determine the likelihood of plants?
- How can we bring together various perspectives to identify the plant growing cycle?

## Background
The plants are classified by the number of growing seasons required to complete their life cycle. There are three main categories: annuals, biennials and perennials. Annuals provide continues blooms throughout the growing season, while biennials provide blooms during their second year of growth. Perennials will bloom for 2 to 8 weeks or longer, however, bloom time will vary and can occur during the beginning, middle, or end of the growing season.

Source: [PennState Extension](https://extension.psu.edu/plant-life-cycles)

According to the Food Agriculture Organization statistical database (FAOSTAT), in 2019, there were 16 most important plant species, without which human existence would be completely vulnerable. The importance of these crops was ranked based on the total production of output (tonnes).

### Most important plants for humankind

1. Sugarcane (annual)
2. Maize (annual)
3. Paddy (annual)
4. Wheat (annual)
5. Potatoes (annual)
6. Soybeans (biennial)
7. Oil palm fruit (perennial)
8. Cassava
9. Tomatoes (perennial)
10. Banana (perennial)
11. Sweet potatoes
12. Yams
13. Sorghum
14. Olive (perennial)
15. Natural rubber (perennial)
16. Coffee (perennial)

Source [Science Agriculture](https://scienceagri.com/16-most-important-plant-species-in-the-world/) ; [FAOSTAT](https://www.fao.org/faostat/en/#data/QCL)

## Vulnerability of cultives to climate 

### Perennial crops
Winkler *et al* states that the vulnerability of perennial crops has not been investigated as much as that of annual crops. Mainly, because the perennial crop production is frequently constrained to areas with unique local or regional climates that are often influenced by topographic position, proximity to water bodies, or the presence of mesoscale atmospheric circulations and weather conditions such as frequent fog that modify the local climate.

> The fruit crops most investigated are wine grapes, apple, cherry, citrus, peach, apricot, kiwi, mango, pear, pineapple, plum, and strawberry.

Perennial crops are particularly vulnerable to cold damage at three distinct stages:

1. In the fall before the tree is adequately hardened.
2. During the winter dormant period when severe cold events can cause injury to woody tissue
3. During spring when temperatures slightly below freezing may kill flower buds following the loss of cold hardiness

*(Raseira and Moore 1987)*.

There are other problems that perennial crops have to face as a consequence of climate change, among which we can find winter chill fulfillment, springtime freeze risk, pollination constraints, heat stress, disease and insect pest damage.

Source: [Winkler et al](https://www.sciencedirect.com/science/article/pii/B9780123847034002082)

### Annuals crops
Temperature is a major determinant of the rate of plant development and, under climate change, warmer temperatures that shorten development stages of determinate crops will most probably reduce the yield of a given variety. In that sense, crop production is inherently sensitive to variability in climate. For example, the yield of wheat declined by 5–8% per 1°C rise in mean seasonal temperature. Also, earlier crop flowering and maturity have been observed and documented in recent decades, and these are often associated with warmer (spring) temperatures.

Source: [Craufurd and Wheeler](https://pubmed.ncbi.nlm.nih.gov/19505929/#:~:text=Crop%20production%20is%20inherently%20sensitive,yield%20of%20a%20given%20variety.)

> Annual crops as maize, rice, wheat and soybeans, together account for 75% of global dietary energy intake *(Cassman, 1999)*. <br>
> Perennial grains, legumes and oilseed varieties represent a paradigm shift in modern agriculture and hold great potential for truly sustainable production systems. [The Land Institute](https://landinstitute.org/our-work/perennial-crops/)


## Growth phases
We must first define the growth stages of the maize crop we are analysing. There are several methods for the identification of the stages in which two methods stand out The ''collar'' and The ''droopy leaf'' method Source: [Heidi Reed](https://extension.psu.edu/corn-growth-stages#:~:text=The%20two%20phases%20of%20corn,every%203%20to%204%20days%%20.)


These methods consist of counting leaves in order to know the age of the corn. Due to the quality and size of the satellite photos it is very difficult to count the number of leaves of the corn, so to define its growth phases the area of interest for the analysis are:
 - **Emergence**.Time when seeds have been sprouted.
 - **Senescence**.When the crop reaches adulthood; starts to turn brown.
 - **Harvest**.When the field is harvested.
 
Source [De Castro *et al*](https://www.mdpi.com/2072-4292/10/11/1745)

## What has been done?

Most of the empirical climate change economics literature uses cross-section/time-series econometrics modeling to estimate the response of crop yields to weather shocks. One exception is the employ of long-differences approach to highlight that adjustments of maize and soybean cultivation in the United States (US) over time-frames of decades or more have resulted in only modest attenuation of the adverse effects of extreme heat exposures on yield losses.

Source: [Wing, De Cian, Mistry](https://www.sciencedirect.com/science/article/pii/S0095069621000450)


### Images detection

It is possible to detect the products grown on farmland before the end of the growing season using time series of satellite images and with different classification models such as random forest and support vector machine. For this, it is necessary to know the availability in precise spaces and in real time on the farmland. This information is obtained through satellite images that cover the crop field in past seasons and current time to generate planning based on data.

Source: [Rahmati, Zoej, Dehkordi](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0273117722004173)

### An intelligent system for crop identification and classification from UAV images using conjugated dense convolutional neural network

These papers uses RGB images obtained by UAV to elucidate a new conjugated dense CNN (CD-CNN) architecture with a new activation function named SL-ReLU for intelligent classification of multiple crops. 

The principals constraints that these method has are: it does not use fully connected network, the used images has a size of 5472 × 3648 pixels, the UAV flight height was about 100 meter above the ground, the front and side overlaps were fixed to 75% and 70% respectively and the crop region in a single raw image contains only a single type of crop and other portion of the raw image is non-crop region. Finally, the proposed CNN module eliminates the use of separate feature extraction methods for learning features from candidate crop regions. The training time is significantly lesser from standard CNN, like ResNet-50, due to lesser depth of the CD-CNN and absence of fully connected layers. The experiment is carried out for a data set of five different crops and it is shown that the proposed module achieves an accuracy of 96.2%, which proves itself superior to any other standard multi-class classification techniques.


Source [Pandey & Kamal](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169921005603)

### Automatic expert system for weeds/crops identification in images from maize fields

Most existing strategies address the problem of green identification under the assumption that plants display a high degree of greenness, but they do not consider the fact that plants may have lost their degree of greenness for different reasons.

Source [Montalvo *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0957417412008895)

### Identification of cash crop diseases using automatic image segmentation algorithm and deep learning with expanded dataset

Early identification of diseases and pests in a field is vital in the production of any crop. This is because late detection could mean that a large part of the crop has been infected which reduces the quality of the crop. This paper presents a more efficient alternative to GrabCut used for image segmentation. This alternative is the AISA algorithm which preprocesses the images at a lower time cost. In addition, a CNN was used for deep learning; where it is trained with public data from PlantVillage (Plant disease database). Finally, a first approach to the use of smartphones is presented so that with their camera they are able to monitor the crop, allowing people with fields to reduce costs and make these tools more accessible. Overall the model was 84.83% accurate in identifying crop diseases.

Source [Xiong *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169920300284)

### Crop disease identification and interpretation method based on multimodal deep learning
This article deals with the identification of common invasive diseases in tomato and cucumber. A multimodal data analysis is presented in which the literature is examined and what it says about the diseases in order to find a correlation between the image and the text. Afterwards, the identification analysis was made to be reliable, i.e. the image identification learning was effective. Finally, an "image-text" disease identification model based on the ITK-Net model is presented. The model presented achieved an accuracy of 99.63%, 99%, 99.07% and 99.78% depending on the database used. In this way, the analysis for the identification of diseases is becoming more and more complete.

Source [Zhou *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169921004257)
### Analysis of clustering methods for crop type mapping using satellite imagery
The objective of this article was to compare algorithms in order to see which algorithm is more efficient when trained with a specific crop to calculate NDVI values and, given an unknown crop, to give a good result. For this purpose, 4 algorithms were compared in which 5 different crops and 26 datasets with more than 45000 plots were analysed. With these plots we searched for the best clustering algorithms for all types of crops. Where the best option varies depending on the distances of one crop from another in order to make a comparison. In this way they were classified as follows: Manhattan for the DIANA and the PAM, Euclidean and Manhattan for the Spectral and Manhattan and Minkowski for the Agglomerative clustering methods.

Source [Rivera *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0925231222003691)

### Deep Learning model of sequential image classifier for crop disease detection in plantain tree cultivation 
This paper presents a Deep Learning Model aimed at identifying the different diseases of Plantain tree such as Black Sigatoka/Yellow sigatoka, Panama, Bunchy top, Moko, chlorosis, etc. in Tamil Nadu situated in the Southern part of India. This is done by combining . Convolutional Neural Networks (CNN) and Recurrent Neural Network (RNN) which is called as G-Recurrent Convolutional Neural Network (G-RecConNNN) which reduces the pre-processing of the data making it more efficient.

Source [Nandhini *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169922002320)

### Prediction of cotton lint yield from phenology of crop indices using artificial neural networks

In this work, remote sensing is used to determine the colour of cotton in a field. We analysed the growth stages of 2013 and 2014 where the information from the bands (Visible and NearInfrared (NIR)) was obtained from Landsat 8, calculating the vegetation indices (NDVI, GNDVI).  Using ANN, an error of 8% was obtained with values already normalised.
Source [Haghverdi *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169918307166)

## The project as Climate Ai
Climate AI currently has solutions available to its four main types of customers: *Processors and packers*, *Seed input and chemical companies*, *Agriculture Finance* and *Retailers, Co-ops, Crop Advisors*. Some of these products are mainly focused on predictions of the input of the crops, identification of optimal location for selected crops and forecasting of prices and demand of the product. Nevertheless, we identified that just one handles the plants life cycle and, as we believe, cares only about annual crops. 

For these reasons, the objective of this project is the prediction of the annual NDVI curves, which has proven to be a reliable indicator of the development of the life cycle of annual crops. Particularly, special attention is paid to the three phases seen in the previous section. [[Growth Phases](#Growth-phases)].
