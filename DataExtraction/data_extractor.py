import os
import ee
import geetools
import geemap
from dotenv import load_dotenv

# Environment variables.
load_dotenv()

credentials = ee.ServiceAccountCredentials(
    os.environ['SERVICE_ACCOUNT'],
    os.path.join(os.path.dirname(__file__), os.environ['JSON'])
)

ee.Initialize(credentials)


class DataExtractor():
    def __init__(self, data_collection, start_date, end_date, bands):
        self.data_collection = data_collection
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.point = None
        self.region = None
        self.image_collection = None
        self.Map = None

    def set_point(self, point):
        point.reverse()
        self.point = point

    def set_interest_region(self, meters):
        """
        Select an arbitrary point and a distance in (meters),
        to construct a rectangle centered on the given point.
        """
        point = ee.Geometry.Point(self.point)
        self.region = point.buffer(meters).bounds()

        Map = geemap.Map()
        Map.setCenter(self.point[0], self.point[1], zoom=15)
        Map.addLayer(
            point,
            {'color': 'black'},
            'Geometry [black]: point'
        )

        Map.addLayer(self.region)

        self.Map = Map

        return self.Map

    def get_data_visualization(self):
        """
        Get the Earth Engine image collection with the desired characteristics
        """
        self.image_collection = (
            ee.ImageCollection(self.data_collection)
            .select(self.bands)
            .filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
        )

        print(f"""The size of the collection is: {
            self.image_collection.size().getInfo()
            }""")

        visualization = {
            "bands": self.bands[0:3],
            "min": 300,
            "max": 3500
        }

        self.Map.addLayer(
            ee_object=self.image_collection,
            vis_params=visualization,
            name="Sentinel-2",
            shown=False)

        self.Map.add_time_slider(
            ee_object=self.image_collection,
            vis_params=visualization,
            time_interval=5
        )

    def extract_data(self, folder_name="ClimateDate"):
        """
        Iterates over the ImageCollection and send the images to Drive
        """
        for band in self.bands:
            extra = dict(sat="S-HARMONIZED", band=band)
            geetools.batch.Export.imagecollection.toDrive(
                collection=self.image_collection.select(band),
                region=self.region,
                namePattern="{sat}_{id}_{system_date}_{band}",
                datePattern="ddMMMy",
                dataType="int",
                folder=folder_name,
                extra=extra,
                verbose=True
            )
