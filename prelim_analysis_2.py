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

##### PLOT RESULTS - EXAMPLE 2 - INTERACTIVE

# Load final data.
data_final = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_subawards.csv")
)

# Convert geometry string to geometry type.
data_final["geometry"] = geopandas.GeoSeries.from_wkt(data_final["geometry"])

# Create Geopandas Data Frame from final data.
data_final = geopandas.GeoDataFrame(data_final, geometry = "geometry")

# Function that returns a Data Frame of subawardees with given NAICS.
def get_data_final_naics(naics = ""):
    """
    The get_data_final_naics function returns a Data Frame of subawardees with given NAICS.
  
    Arguments
    -----
    naics: string
        A string of which NAICS subawardees to map.

    return
    -----
        A Pandas Data Frame of subawardees with given NAICS.
    """
    
    # Collect data associated with the given NAICS.
    if len(naics) == 2:
        data_final_naics = data_final.query("Code_2 == '" + naics + "'")
    elif len(naics) == 3:
        data_final_naics = data_final.query("Code_3 == '" + naics + "'")
    elif len(naics) == 4:
        data_final_naics = data_final.query("Code_4 == '" + naics + "'")
    elif len(naics) == 5:
        data_final_naics = data_final.query("Code_5 == '" + naics + "'")
    elif len(naics) == 6:
        data_final_naics = data_final.query("Code_6 == '" + naics + "'")
    else:
        data_final_naics = data_final
    
    # Return Data Frame with given NAICS.    
    return(data_final_naics)

# Function that plots the subawards on a global map.
def plot_subaward_map(naics = ""):
    """
    The plot_subaward_map function creates and saves a map for the given NAICS subawardees.
  
    Arguments
    -----
    naics: string
        A string of which NAICS subawardees to map.

    return
    -----
        N/A
    """
    
    # Collect data associated with the given NAICS.    
    data_final_naics = get_data_final_naics(naics)
    
    # Create world layer for map.
    map = folium.Map(location = [data_final_naics.Latitude.mean(), data_final_naics.Longitude.mean()], zoom_start = 4, control_scale = True)

    # Add subaward data to world map.
    data_final_naics.explore(
        m = map, # Pass the map object.
        column = "subaward_amount_avg",
        tooltip = ["subawardee_name", "subaward_amount_avg_str"], # Show custom tooltip.
        name = "subawards" # Name of the layer on the map.
    )

    # Save the map.
    map.save("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/interactive_map.html")

# Run function to plot map and save to HTML.
plot_subaward_map()
