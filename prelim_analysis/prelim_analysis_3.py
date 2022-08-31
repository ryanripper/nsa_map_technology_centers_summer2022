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

##### MACHINE LEARNING

# Load final data.
data_final = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_subawards.csv")
)

# Collect code columns to convert to strings.
code_cols = [col for col in data_final.columns if "Code_" in col]

# Convert each code column to a string.
for col in code_cols:
    data_final[col] = data_final[col].astype("string")
    
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

# Select test subset of all subaward data.
data_final_test = get_data_final_naics("2131")

# Collect distinct Longitude and Latitude of test subset.
coords = data_final_test[["Latitude", "Longitude"]].drop_duplicates()

## Determine optimal epsilon for DBSCAN.

# Instantiate the Nearest Neighbors algorithm on two neighbors.
neigh = NearestNeighbors(n_neighbors = 2)

# Fit the model to the distinct coordinates.
nbrs = neigh.fit(coords)

# Collect the distances and indices of the coordinates once fit.
distances, indices = nbrs.kneighbors(coords)

# Sort the distances in ascending order.
distances = np.sort(distances, axis = 0)

# Collect distances except first one.
distances = distances[:, 1]

# Plot distances.
px.scatter(distances)

## Implement DBSCAN algorithm and plot results.

# Use best epsilon from elbow-plot above.
epsilon = 0.1468624

# Instantiate and fit the DBSCAN model.
db = DBSCAN(eps = epsilon, min_samples = 3, algorithm = "ball_tree", metric = "haversine").fit(np.radians(coords))

# Collect the cluster labels.
cluster_labels = db.labels_

# Identify the number of clusters.
n_clusters = len(set(cluster_labels))

# Create series of clusters from model results.
clusters = pd.Series([coords[cluster_labels == n] for n in range(-1, n_clusters)])

# Add labels to the coordinates.
coords["Cluster"] = db.labels_

# Add the cluster assignment to the original subawardee Data Frame.
data_final_test = data_final_test.merge(coords, on = ["Latitude", "Longitude"], how = "left")

# Create world layer for map.
map = folium.Map(location = [data_final_test.Latitude.mean(), data_final_test.Longitude.mean()], zoom_start = 4, control_scale = True)

# Add subaward data to world map.
data_final_test.explore(
    m = map, # Pass the map object.
    column = "Cluster", # Which column to color.
    tooltip = ["subawardee_name", "subaward_amount_avg_str"], # Show custom tooltip.
    name = "subawards" # Name of the layer on the map.
)

# Save the map.
map.save("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/prelim_analysis/interactive_map.html")
