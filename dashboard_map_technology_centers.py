# Import necessary Python modules.
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import geopandas
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import folium
import mapclassify
import warnings
from kneed import KneeLocator

# Add additional settings to notebook.
warnings.filterwarnings("ignore")

# Include title of page.
st.write("### National Security Agency - Centers of Technology Development by NAICS")

#####

# Load NAICS data.
naics_final = \
(pd
 .read_csv("/Users/ryanripper/Desktop/NSA/nsa_map_technology_centers_summer2022/final/final_naics.csv")
)

# Convert each column to a string.
for col in naics_final.columns:
    naics_final[col] = naics_final[col].astype("string")

# Create new column to hold Title - Code combo.
naics_final["Title_Code"] = naics_final["Title"] + " - (NAICS Code: " + naics_final["Code"] + ")"

# Create empty row for all NAICS.
empty_row = pd.DataFrame({"Code" : [""], "Title" : ["All"], "Title_Code" : ["All"]})

# Combine results.
naics_final = empty_row.append(naics_final.sort_values("Title"), ignore_index = True)

# Collect NAICS Title - Code combos.
title_codes = naics_final.Title_Code

# Create dropdown with NAICS code options.
naics_title = st.selectbox("Select NAICS Code to first evaluate optimal Epsilon value for DBSCAN Machine Learning Algorithm:", title_codes)

# Find associated NAICS code from Title - Code selected.
naics_code = naics_final["Code"].loc[naics_final["Title_Code"] == naics_title].iloc[0]

# Add radio button to ask which ML model to use. Choose DBSCAN to start.
model_type = st.radio("Select ML model to identify clusters", ("DBSCAN", "Optical"), 0)

#####

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

#####

# Function that returns a tuple of coordinates and subawardees with given NAICS for both DBSCAN and Optical and plot to find the optimal epsilon for DBSCAN.
def get_optimal(model_type, naics = ""):
    """
    The get_optimal function returns a tuple of coordinates and subawardees with given NAICS for both DBSCAN and Optical and creates plot to find the optimal epsilon for DBSCAN.
  
    Arguments
    -----
    model_type: string
        A string of which ML model was selected.
    
    naics: string
        A string of which NAICS subawardees to map. Default to no selected NAICS.

    return
    -----
        A tuple of the distinct Longitude and Latitude of test subset and subawardees with given NAICS.
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
        
    # Collect distinct Longitude and Latitude of test subset.
    coords = data_final_naics[["Latitude", "Longitude"]].drop_duplicates()
    
    # Determine optimal epsilon for DBSCAN.
    if model_type == "DBSCAN":
        # Instantiate the Nearest Neighbors algorithm on two neighbors.
        neigh = NearestNeighbors(n_neighbors = 2)

        # Fit the model to the distinct coordinates.
        nbrs = neigh.fit(coords)

        # Collect the distances and indices of the coordinates once fit.
        distances, indices = nbrs.kneighbors(coords)

        # Sort the distances in ascending order.
        distances = np.sort(distances, axis = 0)

        # Collect distances only.
        distances = distances[:, 1]

        # Plot distances.
        fig = px.line(distances, labels = {"index" : "", "value" : ""}, title = "Determine Optimal Epsilon Value for DBSCAN")

        # Remove legend from plot.
        fig.update_layout(showlegend = False,
                          title = {"y" : 0.9, "x" : 0.5, "xanchor" : "center", "yanchor" : "top"}
        )
        
        # Show the plotly chart.
        st.plotly_chart(fig)
        
        # Get the knee from the figure.
        kneedle = KneeLocator(range(1, len(distances) + 1), # x values
                              distances, # y values
                              S = 1.0, # parameter suggested from paper
                              curve = "convex", # parameter from figure
                              direction = "increasing") # parameter from figure
        
        # Get x-coordinate from knee point.
        elbow_point_x = kneedle.elbow
        
        # Get y-coordinate from knee point.
        elbow_point_y = kneedle.elbow_y
        
        # Display the coordinates of the knee.
        st.write(f"Elbow Point: ({elbow_point_x}, {elbow_point_y})")
    
    # Return coordinates and subawardees with given NAICS.
    return(coords, data_final_naics)

#####

# If ML model chosen is DBSCAN.
if model_type == "DBSCAN":
    # Run function to get results.
    optimal_results = get_optimal(model_type, naics_code)

    # Identify coordinates from results.
    coords = optimal_results[0]

    # Select test subset of all subaward data and plot optimal epislon for DBSCAN.
    data_final_test = optimal_results[1]

    # Enter value for epislon
    epsilon = st.number_input("Enter Epsilon Value for DBSCAN at the Point of Maximum Curvature:")

    # Only continue if epsilon has been input.
    if epsilon != 0:
        # Instantiate and fit the DBSCAN model.
        db = DBSCAN(eps = epsilon, min_samples = 3, algorithm = "ball_tree", metric = "haversine").fit(np.radians(coords))
        
        # Identify the unique cluster labels.
        cluster_labels = set(db.labels_)
        
        # Remove the outlier from the cluster labels.
        cluster_labels.discard(-1)

        # Identify the number of clusters without outlier.
        n_clusters = len(cluster_labels)

        # Show how many clusters were identified.
        st.write(f"Number of Clusters (Without Outliers): {n_clusters}")

        # Add labels to the coordinates.
        coords["Cluster"] = db.labels_

        # Add the cluster assignment to the original subawardee Data Frame.
        data_final_test = data_final_test.merge(coords, on = ["Latitude", "Longitude"], how = "left")

        # Create world layer for map.
        map = folium.Map(control_scale = False, zoom_start = 1, width = 725, height = 500)

        # Add subaward data to world map.
        data_final_test.explore(
            m = map, # Pass the map object.
            column = "Cluster", # Which column to color.
            tooltip = False, # ["subawardee_name", "subaward_amount_avg_str"], # Show custom tooltip.
            legend = False,
            name = "subawards", # Name of the layer on the map.
            cmap = "Set1" # Color the clusters.
        )
        
        # Collect the average location of each cluster.
        cluster_results = \
        (data_final_test
         .groupby(["Cluster"])
         .mean()
         .reset_index()
        )
        
        # Get the number of subawards in each cluster.
        cluster_results_count = \
        (data_final_test
         .groupby(["Cluster"])
        )["Cluster"].count()
        
        # Make number of subawards in each cluster into a Data Frame.
        cluster_results_count = \
        (pd
         .DataFrame({"Cluster" : cluster_results_count.index, "Count" : cluster_results_count.values})
        )
        
        # Merge the average location and number of subawards in each cluster into one Data Frame.
        cluster_results = \
        (cluster_results
         .merge(cluster_results_count, how = "inner", on = "Cluster")
        )
        
        # Convert cluster average subaward amount to string.
        cluster_results["Cluster Average Subaward Amount"] = \
        "$" + \
        (cluster_results["subaward_amount_avg"]
         .astype(int)
         .map("{:,}".format)
        )
        
        # Format count with commas.
        cluster_results["Number of Subawardees in Cluster"] = \
        (cluster_results["Count"]
         .astype(int)
         .map("{:,}".format)
        )
        
        # Convert cluster data into a Geopandas Data Frame and remove outlier cluster group (now 0).
        cluster_results_test = geopandas.GeoDataFrame(cluster_results, geometry = geopandas.points_from_xy(cluster_results.Longitude, cluster_results.Latitude)).query("Cluster != -1")
        
        # Add cluster data to world map.
        cluster_results_test.explore(
            m = map, # Pass the map object.
            tooltip = ["Cluster Average Subaward Amount", "Number of Subawardees in Cluster"], # Show custom tooltip.
            color = "black" # Color the points.
        )
        
        # Map everything to the page.
        st_data = st_folium(map, width = 725, height = 500)
# If ML model chosen is OPTICAL.
elif model_type == "Optical":
    # Run function to get results.
    optimal_results = get_optimal(model_type, naics_code)

    # Identify coordinates from results.
    coords = optimal_results[0]
    
    # Select test subset of all subaward data and plot optimal epislon for DBSCAN.
    data_final_test = optimal_results[1]
    
    # Instantiate and fit the OPTICS model.
    optics = OPTICS(min_samples = 3).fit(np.radians(coords))
    
    # Identify the unique cluster labels.
    cluster_labels = set(optics.labels_)
    
    # Remove the outlier from the cluster labels.
    cluster_labels.discard(-1)

    # Identify the number of clusters without outlier.
    n_clusters = len(cluster_labels)

    # Show how many clusters were identified.
    st.write(f"Number of Clusters (Without Outliers): {n_clusters}")

    # Add labels to the coordinates.
    coords["Cluster"] = optics.labels_

    # Add the cluster assignment to the original subawardee Data Frame.
    data_final_test = data_final_test.merge(coords, on = ["Latitude", "Longitude"], how = "left")

    # Create world layer for map.
    map = folium.Map(control_scale = False, zoom_start = 1, width = 725, height = 500)

    # Add subaward data to world map.
    data_final_test.explore(
        m = map, # Pass the map object.
        column = "Cluster", # Which column to color.
        tooltip = False, # ["subawardee_name", "subaward_amount_avg_str"], # Show custom tooltip.
        legend = False,
        name = "subawards", # Name of the layer on the map.
        cmap = "Set1" # Color the clusters.
    )
    
    # Collect the average location of each cluster.
    cluster_results = \
    (data_final_test
     .groupby(["Cluster"])
     .mean()
     .reset_index()
    )
    
    # Get the number of subawards in each cluster.
    cluster_results_count = \
    (data_final_test
     .groupby(["Cluster"])
    )["Cluster"].count()
    
    # Make number of subawards in each cluster into a Data Frame.
    cluster_results_count = \
    (pd
     .DataFrame({"Cluster" : cluster_results_count.index, "Count" : cluster_results_count.values})
    )
    
    # Merge the average location and number of subawards in each cluster into one Data Frame.
    cluster_results = \
    (cluster_results
     .merge(cluster_results_count, how = "inner", on = "Cluster")
    )
    
    # Convert cluster average subaward amount to string.
    cluster_results["Cluster Average Subaward Amount"] = \
    "$" + \
    (cluster_results["subaward_amount_avg"]
     .astype(int)
     .map("{:,}".format)
    )
    
    # Format count with commas.
    cluster_results["Number of Subawardees in Cluster"] = \
    (cluster_results["Count"]
     .astype(int)
     .map("{:,}".format)
    )
    
    # Convert cluster data into a Geopandas Data Frame and remove outlier cluster group (now 0).
    cluster_results_test = geopandas.GeoDataFrame(cluster_results, geometry = geopandas.points_from_xy(cluster_results.Longitude, cluster_results.Latitude)).query("Cluster != -1")
    
    # Add cluster data to world map.
    cluster_results_test.explore(
        m = map, # Pass the map object.
        tooltip = ["Cluster Average Subaward Amount", "Number of Subawardees in Cluster"], # Show custom tooltip.
        color = "black" # Color the points.
    )
    
    # Map everything to the page.
    st_data = st_folium(map, width = 725, height = 500)