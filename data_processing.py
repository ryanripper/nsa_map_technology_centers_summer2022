# Import necessary Python modules.
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import missingno as miss
from plotnine import *
import matplotlib.pyplot as plt
import geopandas
import folium
import mapclassify
import numpy as np
from geopy.geocoders import Nominatim
import warnings
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# Add additional settings to notebook.
warnings.filterwarnings("ignore")

##### DATA COLLECTION AND WRANGLING

# Read in all subaward data.
data_sub_awards_all = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/raw/All_Contracts_Subawards_2022-04-12_H19M40S39_1.csv")
)

# Create column with first two numbers of NIACS.
data_sub_awards_all["prime_award_naics_code_2"] = data_sub_awards_all.prime_award_naics_code.astype("string").str[:2]

# Create column with first three numbers of NIACS.
data_sub_awards_all["prime_award_naics_code_3"] = data_sub_awards_all.prime_award_naics_code.astype("string").str[:3]

# Create column with first four numbers of NIACS.
data_sub_awards_all["prime_award_naics_code_4"] = data_sub_awards_all.prime_award_naics_code.astype("string").str[:4]

# Create column with first five numbers of NIACS.
data_sub_awards_all["prime_award_naics_code_5"] = data_sub_awards_all.prime_award_naics_code.astype("string").str[:5]

# Create column with first six numbers of NIACS.
data_sub_awards_all["prime_award_naics_code_6"] = data_sub_awards_all.prime_award_naics_code.astype("string").str[:6]

# Load 2-6 digit 2022 Code file for NAICS. https://www.census.gov/naics/?48967
naics_all = \
(pd
 .read_excel("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/raw/2-6 digit_2022_Codes.xlsx")
 .iloc[:, [1, 2]]
)

# Rename columns for NAICS.
naics_all.columns = (naics_all
                     .columns
                     .str
                     .strip()
                     .str
                     .replace(" ", "_")
                     .str
                     .lower()
                    )

# Rename the columns again for more accessibility.
naics_all = naics_all.rename(columns = {"2022_naics_us___code" : "Code", "2022_naics_us_title" : "Title"})

# Change all columns to object type.
naics_all = naics_all.astype("string")

# Join on every possible 2-6 digit 2022 code from the LHS.
data_sub_awards_all_naics = \
(data_sub_awards_all
 .merge(naics_all.loc[naics_all.Code.str.len() == 2], left_on = "prime_award_naics_code_2", right_on = "Code")
 .rename(columns = {"Code" : "Code_2", "Title" : "Title_2"})
 .merge(naics_all.loc[naics_all.Code.str.len() == 3], left_on = "prime_award_naics_code_3", right_on = "Code")
 .rename(columns = {"Code" : "Code_3", "Title" : "Title_3"})
 .merge(naics_all.loc[naics_all.Code.str.len() == 4], left_on = "prime_award_naics_code_4", right_on = "Code")
 .rename(columns = {"Code" : "Code_4", "Title" : "Title_4"})
 .merge(naics_all.loc[naics_all.Code.str.len() == 5], left_on = "prime_award_naics_code_5", right_on = "Code")
 .rename(columns = {"Code" : "Code_5", "Title" : "Title_5"})
 .merge(naics_all.loc[naics_all.Code.str.len() == 6], left_on = "prime_award_naics_code_6", right_on = "Code")
 .rename(columns = {"Code" : "Code_6", "Title" : "Title_6"})
)

# Create address for each subaward recipient.
data_sub_awards_all_naics["subawardee_address"] = \
(data_sub_awards_all_naics[["subawardee_address_line_1", "subawardee_city_name"]]
 .fillna("")
 .astype("string")
)[["subawardee_address_line_1", "subawardee_city_name"]].agg(", ".join, axis = 1) # + ", " + \
# (data_sub_awards_all_naics
 # .subawardee_state_code
 # .fillna("")
 # .astype("string")
# ) + " " + \
# (data_sub_awards_all_naics
 # .subawardee_zip_code
 # .fillna("")
 # .astype("string")
 # .str[:6]
# )

# Get distinct addresses.
subawardee_address = pd.DataFrame({"Address" : data_sub_awards_all_naics.subawardee_address.unique()})

# Define a function that takes all contract addresses and returns location information.
def get_locations_results(addresses):
    """
    The get_locations_results function finds the location information for a dataframe of addresses and returns a dataframe with all location information.
  
    Arguments
    -----
    addresses: Pandas DataFrame
        A DataFrame of all addresses.

    return
    -----
        A Pandas DataFrame with each corresponding address location information.
    """
    
    # Initialize Nominatim API.
    geolocator = Nominatim(user_agent = "NAICS_Address")

    # Create empty dictionary to hold locations.
    dict_locations = {}

    # For each unique address.
    for loc in addresses.Address:
        # Get location of address.
        location = geolocator.geocode(loc, timeout = None)
        
        # Add to location dictionary only when a location is found.
        if location is not None:
            dict_locations[loc] = (location.address, location.latitude, location.longitude)

    # Push dictionary results into a dataframe.
    locations_results = pd.DataFrame([[k, v[0], v[1], v[2]] for k, v in dict_locations.items() if v != "ERROR"]).rename(columns = {0 : "Address", 1 : "Location", 2 : "Latitude", 3 : "Longitude"})

    # Return results.
    return(locations_results)

# Save results to a CSV to avoid having to run again.
# get_locations_results(subawardee_address).to_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/subawardee_locations.csv", index = False)

# Load the address results.
locations_results = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/subawardee_locations.csv")
)

# Merge address results to main contract dataframe.
data_sub_awards_all_naics_final = \
(data_sub_awards_all_naics
 .merge(locations_results, left_on = "subawardee_address", right_on = "Address", how = "inner")
)

# Find average contract subaward amount for each subawardee.
data_sub_awards_all_naics_final["subaward_amount_avg"] = \
(data_sub_awards_all_naics_final["subaward_amount"]
 .groupby(data_sub_awards_all_naics_final["subawardee_name"])
 .transform("mean")
)

# Collect columns of interest, drop all missing values, and drop duplicate rows.
data_final = \
data_sub_awards_all_naics_final[[
    "subaward_amount_avg",
    "subawardee_name",
    "subawardee_address",
    "Address",
    "Location",
    "Latitude",
    "Longitude",
    "prime_award_naics_code",
    "Code_2",
    "Title_2",
    "Code_3",
    "Title_3",
    "Code_4",
    "Title_4",
    "Code_5",
    "Title_5",
    "Code_6",
    "Title_6"
]].dropna().drop_duplicates().sort_values(by = ["subawardee_name"])

# Convert average subaward amount to string.
data_final["subaward_amount_avg_str"] = \
"$" + \
(data_final["subaward_amount_avg"]
 .astype(int)
 .map("{:,}".format)
)

# Convert DataFrame to GeoDataFrame.
data_final = geopandas.GeoDataFrame(data_final, geometry = geopandas.points_from_xy(data_final.Longitude, data_final.Latitude))

# Save final data to CSV for future use.
# data_final.to_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_subawards.csv", index = False)

# Create Data Frame to hold all possible NAICS values.
naics_options = pd.concat([
    data_final[["Code_2", "Title_2"]].rename(columns = {"Code_2" : "Code", "Title_2" : "Title"}),
    data_final[["Code_3", "Title_3"]].rename(columns = {"Code_3" : "Code", "Title_3" : "Title"}),
    data_final[["Code_4", "Title_4"]].rename(columns = {"Code_4" : "Code", "Title_4" : "Title"}),
    data_final[["Code_5", "Title_5"]].rename(columns = {"Code_5" : "Code", "Title_5" : "Title"}),
    data_final[["Code_6", "Title_6"]].rename(columns = {"Code_6" : "Code", "Title_6" : "Title"})
], ignore_index = True).drop_duplicates().sort_values(by = ["Code"])

# Save final NAICS data to CSV for future use.
# naics_options.to_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_naics.csv", index = False)