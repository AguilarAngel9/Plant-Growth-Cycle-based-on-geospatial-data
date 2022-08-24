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

Perennial crops are particularly vulnerable to cold damage at three distinct stages: **(1) in the fall before the tree is adequately hardened**, **(2) during the winter dormant period when severe cold events can cause injury to woody tissue**, and **(3) during spring when temperatures slightly below freezing may kill flower buds following the loss of cold hardiness** *(Raseira and Moore 1987)*.

There are other problems that perennial crops have to face as a consequence of climate change, among which we can find winter chill fulfillment, springtime freeze risk, pollination constraints, heat stress, disease and insect pest damage.

Source: [Winkler et al](https://www.sciencedirect.com/science/article/pii/B9780123847034002082)

### Annuals crops
Temperature is a major determinant of the rate of plant development and, under climate change, warmer temperatures that shorten development stages of determinate crops will most probably reduce the yield of a given variety. In that sense, crop production is inherently sensitive to variability in climate. For example, the yield of wheat declined by 5–8% per 1°C rise in mean seasonal temperature. Also, earlier crop flowering and maturity have been observed and documented in recent decades, and these are often associated with warmer (spring) temperatures.

Source: [Craufurd and Wheeler](https://pubmed.ncbi.nlm.nih.gov/19505929/#:~:text=Crop%20production%20is%20inherently%20sensitive,yield%20of%20a%20given%20variety.)

> Annual crops as maize, rice, wheat and soybeans, together account for 75% of global dietary energy intake *(Cassman, 1999)*. <br>
> Perennial grains, legumes and oilseed varieties represent a paradigm shift in modern agriculture and hold great potential for truly sustainable production systems. [The Land Institute](https://landinstitute.org/our-work/perennial-crops/)


## Growth phases
At the moment of having a crop we have to analyze its growth through the phases of growth that it has. These phases are considered already once the seed is sown, that is, already when the plant was born / transplanted in the land of interest. We have that the growth phases are divided into three: Rapid growth, Hardening and Establishment phase.

- **Establishment phase**. Duration of 14 to 21 days to germinate and 4 to 8 weeks of initial growth. It is where the seed is just planted and is expected to have the first sprout, where we first ensure that the roots develop properly in case of being in a container and then be transplanted.
- **Rapid growth**. Varies according to the crop and the conditions it goes around 8 to 20 weeks. As it indicates, the plants (buds) grow in a very fast way, this from nutrients or chemicals applied.
- **Hardening**. 1 to 4 months depending on the crop. This is where the energy of the plant is focused on the growth of the stem and roots so that it has a firm foundation and has a better chance of survival. In this phase the plants are prepared to withstand different conditions (withstand stress).

TO DO: Merge our stages with the ones used in the company.

Besides this, it is important to consider the factors of irrigation, fertilization, spraying and temperature that the crop is receiving.

Source [Fuentes de ortiz]()

## What has been done?

Most of the empirical climate change economics literature uses cross-section/time-series econometrics modeling to estimate the response of crop yields to weather shocks. One exception is the employ of long-differences approach to highlight that adjustments of maize and soybean cultivation in the United States (US) over time-frames of decades or more have resulted in only modest attenuation of the adverse effects of extreme heat exposures on yield losses.

Source: [Wing, De Cian, Mistry](https://www.sciencedirect.com/science/article/pii/S0095069621000450)


### Images detection

It is possible to detect the products grown on farmland before the end of the growing season using time series of satellite images and with different classification models such as radom forest and super vector machine. For this, it is necessary to know the availability in precise spaces and in real time on the farmland. This information is obtained through satellite images that cover the crop field in past seasons and current time to generate planning based on data.

Source: [Rahmati, Zoej, Dehkordi](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0273117722004173)

### An intelligent system for crop identification and classification from UAV images using conjugated dense convolutional neural network

These papers uses RGB images obtained by UAV to elucidate a new conjugated dense CNN (CD-CNN) architecture with a new activation function named SL-ReLU for intelligent classification of multiple crops. 

The principals constraints that these method has are: it does not use fully connected network, the used images has a size of 5472 × 3648 pixels, the UAV flight height was about 100 meter above the ground, the front and side overlaps were fixed to 75% and 70% respectively and the crop region in a single raw image contains only a single type of crop and other portion of the raw image is non-crop region. Finally, the proposed CNN module eliminates the use of separate feature extraction methods for learning features from candidate crop regions. The training time is significantly lesser from standard CNN, like ResNet-50, due to lesser depth of the CD-CNN and absence of fully connected layers. The experiment is carried out for a data set of five different crops and it is shown that the proposed module achieves an accuracy of 96.2%, which proves itself superior to any other standard multi-class classification techniques.


Source [Pandey & Kamal](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169921005603)

### Automatic expert system for weeds/crops identification in images from maize fields

Most existing strategies address the problem of green identification under the assumption that plants display a high degree of greenness, but they do not consider the fact that plants may have lost their degree of greenness for different reasons.

Source [Montalvo *et al*](https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0957417412008895)

Papers to read: 
- https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169920300284
- https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169921004257
- https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0925231222003691
- https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169922002320
- https://0-www-sciencedirect-com.biblioteca-ils.tec.mx/science/article/pii/S0168169918307166

## Our buyer
Find.

## The project as Climate Ai
Climate AI currently has solutions available to its four main types of customers: *Processors and packers*, *Seed input and chemical companies*, *Agriculture Finance* and *Retailers, Co-ops, Crop Advisors*. Some of these products are mainly focused on predictions of the input of the crops, identification of optimal location for selected crops and forecasting of prices and demand of the product. Nevertheless, we identified that just one handles the plants life cycle and, as we believe, cares only about annual crops. Meanwhile, as *Winkler (2022)* states climate assessments for perennial crops heavily rely on empirical relationships developed between climate observations and plant phenology, and less frequently, between climate observations and yield.

The project will be focused on the identification and monitoring of the perennial crop development through the lifecycle, giving special attention to the three phases seen in previously section [[Growth Phases](#Growth-phases)].
