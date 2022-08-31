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

##### PLOT RESULTS - EXAMPLE 1 - NON-INTERACTIVE

# Load final data.
data_final = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_subawards.csv")
)

# Load map of the United States. https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
states = geopandas.read_file("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/raw/cb_2018_us_state_500k.shp")

# Load map of the world.
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# Plot the map of the United States.
world.boundary.plot(figsize = (50, 50))

# Plot the points for each subaward.
plt.scatter(data_final["Longitude"], data_final["Latitude"])
