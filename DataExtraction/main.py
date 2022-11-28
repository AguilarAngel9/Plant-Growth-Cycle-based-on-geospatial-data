from typing import Union, Tuple, List
from data_extractor import DataExtractor
from google_drive import retrieve_data_from_Drive, see_data_from_Drive
from copy import deepcopy


def main(
    location: Tuple[float, float],
    time_window: Union[Tuple[str, str], None],
    data_collection: str = 'COPERNICUS/S2_SR_HARMONIZED',
    bands: List[str] = ['B4', 'B3', 'B2', 'B8'],
    folder_name: str = None,
    radius: int = 400
):
    start_date = time_window[0] if time_window else '2019-04-01'
    end_date = time_window[1] if time_window else '2019-12-01'

    # Call the DataExtractor class.
    D = DataExtractor(
        data_collection=data_collection,
        start_date=start_date,
        end_date=end_date,
        bands=bands
    )

    center_of_field = deepcopy(location)

    # Select which location to study.
    D.set_point(center_of_field)

    # Set the radius of the region of interest.
    D.set_interest_region(radius)

    # Get the bands and the visualizations.
    D.get_data_visualization()

    # Send the data to a Google Drive folder.
    D.extract_data(folder_name)

    folder = folder_name if folder_name else 'ClimateDate'

    # Download the data from Drive folder.
    retrieve_data_from_Drive(folder)

if __name__ == "__main__":
    print ("Executed when invoked directly")
    main(
        location=[42.466952550670946, -88.2157321904346],
        time_window=('2019-04-01', '2019-12-01'),
        folder_name='History'
    )