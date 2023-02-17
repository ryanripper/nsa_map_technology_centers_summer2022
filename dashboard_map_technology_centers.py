# Import necessary Python modules.
import streamlit as st
from streamlit_folium import folium_static
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
st.markdown("### National Security Agency Cybersecurity Collaboration Center: Identifying Centers of Technology Sectors Across the Globe")

# Include by line.
st.markdown(r'''
    <h5>Ryan Ripper<br/>Summer 2022 - Georgetown University</h5>
    ''', unsafe_allow_html = True)

#####

# Include header for section.
st.markdown("#### 1. Introduction")

# Style header.
st.markdown(
    """
	<style>
	.streamlit-expanderHeader {
	    font-size: large;
	}
	</style>
	""",
    unsafe_allow_html = True,
)

with st.expander("Motivation", False):
	st.markdown(r'''
	    This dashboard serves as a tool to identify centers of development across different technology sectors around the globe.
	    ''')

with st.expander("Data Source", False):
	st.markdown(r'''
	    All contracts that have been awarded by the United States Government for the calendar year 2021 have been loaded into this dashboard. These contracts are made available by USAspending which is the official open data source of federal spending information.  Data collection and processing has been completed using Python scripts that are available within the author's [Github Repository](https://github.com/ryanripper/nsa_map_technology_centers_summer2022).
	    ''')

	st.markdown(r'''
	    The user has the option to consider all contracts across all technology sectors or to subset these contracts according to an awardee's respective North American Industry Classification System (NAICS) Code.
	    ''')
		
with st.expander("Machine Learning", False):
	st.markdown(r'''
	    The dashboard uses two distinct Machine Learning algorithms to identify groups of awardees within a particular technology sector. Those algorithsm are [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) (Density-Based Spatial Clustering of Applications with Noise) and [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html) (Ordering Points To Identify the Clustering Structure). Both take advantage of the Python module "scikit-learn" to find core samples of high density and expand clusters from them. We leverage the physical location of each awardee based on given Latitude and Longitude as our unlabeled set of data to feed into our algorithms in identifying geographic cluster.
	    ''')
	    
	st.markdown(r'''
	    When the user selects DBSCAN as their machine learning algorithm of choice, the dashboard assists the user in identifying the algorithm's associated hyperparameter, "epsilon." Epsilon is defined as the maximum distance between two samples for one to be considered as in the neighborhood of the other. Figure 1 assists in determining a suitable value for epsilon by calculating the distance to the nearest $n$ points for each point (using Haversine distance), sorting, and plotting the results. The optimal value for epsilon will be found at the point of maximum curvature where the user can select the automated optimal value of epsilon or choose their own value and observe how the clusters are identified by DBSCAN in Figure 2A.
	    ''')
	    
	st.markdown(r'''
	    The OPTICS machine learning algorithm does not rely on user-tuned hyperparameters. Instead, the algorithm keeps cluster hierarchy for a variable neighborhood radius. This functionality automatically identifies the clusters in Figure 2B.
	    ''')
	    
with st.expander("Output", False):
	st.markdown(r'''
	    Figures 2A and 2B display the results of the machine learning clustering algorithms. The map displays each awardee from the subset of total contracts according to the selected NAICS code. Each awardee is colored according to the corresponding cluster group identified by the respective algorithm.
	    ''')

#####

# Include header for section.
st.markdown("#### 2. Select NAICS Code and Machine Learning Algorithm")

# Cached function that loads the required data and transforms for subsequent use within dashboard.
@st.cache_data(allow_output_mutation = True)
def load_data():
	"""
	The load_data function returns a tuple of loaded data to be used throughout the dashboard..

	Arguments
	-----
	NA

	return
	-----
	    A tuple of the final NAICS dataframe, the titles associated with each NAICS code in a dataframe, and the GeoPandas dataframe of all contract awards.
	"""
	
	# Load NAICS data.
	naics_final = \
	(pd
	 .read_csv("./final/final_naics.csv")
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
	
	# Load final data.
	data_final = \
	(pd
	 .read_csv("./final/final_subawards.csv")
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
	
	# Return all necessary data objects.
	return(naics_final, title_codes, data_final)
	
# Load data into streamlit app.
naics_final, title_codes, data_final = load_data()

# Create dropdown with NAICS code options.
naics_title = st.selectbox("Select NAICS Code to Subset All Awarded Government Contracts in CY 2021:", title_codes)

# Find associated NAICS code from Title - Code selected.
naics_code = naics_final["Code"].loc[naics_final["Title_Code"] == naics_title].iloc[0]

# Add radio button to ask which ML model to use. Choose DBSCAN to start.
model_type = st.radio("Select Machine Learning Model to Identify Technology Clusters for Selected Subset:", ("DBSCAN", "OPTICS"), 0)

#####

# A cached function that returns a tuple of coordinates, subawardees with given NAICS, distance between awards, and the point of maximum curvature for the associated elbow plot for both DBSCAN and OPTICS.
@st.cache_data(allow_output_mutation = True)
def get_optimal(data_final, model_type, naics = ""):
	"""
	The get_optimal function returns a tuple of coordinates, subawardees with given NAICS, distance between awards, and the point of maximum curvature for the associated elbow plot for both DBSCAN and OPTICS.

	Arguments
	-----
	data_final: dataframe
		A dataframe of all award contracts.
	
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
	    neigh = NearestNeighbors(n_neighbors = 2, metric = "haversine")

	    # Fit the model to the distinct coordinates.
	    nbrs = neigh.fit(coords)

	    # Collect the distances and indices of the coordinates once fit.
	    distances, indices = nbrs.kneighbors(coords)

	    # Sort the distances in ascending order.
	    distances = np.sort(distances, axis = 0)

	    # Collect distances only. Convert to km from Haversine distance.
	    distances = distances[:, 1] * 6371
	    
	    # Get the knee from the figure.
	    kneedle = KneeLocator(range(1, len(distances) + 1), # x values
	                          distances, # y values
	                          S = 1.0, # parameter suggested from paper
	                          curve = "convex", # parameter from figure
	                          direction = "increasing") # parameter from figure
	    
	    # Get y-coordinate of elbow point.
	    elbow_point_y = round(kneedle.elbow_y, 4)
	# Results for OPTICS.
	else:
		# Null value for distances.
		distances = None
		
		# Null value for elbow point.
		elbow_point_y = None
	
	# Return objects with given NAICS.
	return(coords, data_final_naics, distances, elbow_point_y)
	
# If ML model chosen is DBSCAN.
if model_type == "DBSCAN":
    # Include header for section.
	st.markdown("#### 3. Parameter Results and Selection for DBSCAN")
	
	# Run function to get results.
	coords, data_final_test, distances, elbow_point_y = get_optimal(data_final, model_type, naics_code)
	
	# Plot distances.
	fig = px.line(distances, labels = {"index" : "", "value" : "Distance Between Two Points (km)"}, title = "Figure 1. Determine Optimal Epsilon Value for DBSCAN")

	# Remove legend from plot.
	fig.update_layout(showlegend = False, title = {"y" : 0.9, "x" : 0.5, "xanchor" : "center", "yanchor" : "top"})
	
	# Show the plotly chart.
	st.plotly_chart(fig)
	
	# Display the coordinate of the elbow.
	st.markdown(f"Elbow Point (Point of Maximum Curvature in Figure 1): {elbow_point_y} km")

    # Enter value for epsilon. Convert back to Haversine distance.
	epsilon = st.number_input("Enter Epsilon Value (km) for DBSCAN at the Point of Maximum Curvature from Figure 1:", value = elbow_point_y) / 6371

	# Only continue if epsilon has been input.
	if epsilon != 0:
		# Collect the Latitude and Longitude of the contract award data.
		coords = coords[["Latitude", "Longitude"]]
		
	    # Instantiate and fit the DBSCAN model.
		db = DBSCAN(eps = epsilon, min_samples = 3, algorithm = "auto", metric = "haversine").fit(np.radians(coords))
	    
		# Identify the unique cluster labels.
		cluster_labels = set(db.labels_)

		# Remove the outlier from the cluster labels.
		cluster_labels.discard(-1)

		# Identify the number of clusters without outlier.
		n_clusters = len(cluster_labels)

		# Include header for section.
		st.markdown("#### 4. DBSCAN Clustering Results")

		# Show how many clusters were identified.
		st.markdown(f"Number of Clusters (Without Outliers): {n_clusters}")

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
		
		# Add cluster label to dataframe.
		cluster_results["Cluster Label"] = \
		(cluster_results["Cluster"]
		 .astype(int)
		) + 1 

		# Convert cluster data into a Geopandas Data Frame and remove outlier cluster group (now 0).
		cluster_results_test = geopandas.GeoDataFrame(cluster_results, geometry = geopandas.points_from_xy(cluster_results.Longitude, cluster_results.Latitude)).query("Cluster != -1")

		# Add cluster data to world map.
		cluster_results_test.explore(
			m = map, # Pass the map object.
			tooltip = ["Cluster Label", "Cluster Average Subaward Amount", "Number of Subawardees in Cluster"], # Show custom tooltip.
			color = "black" # Color the points.
		)

		# Include title to figure.
		st.markdown(r'''
			<p style="text-align: center;"">Figure 2A. Map of DBSCAN Clusters for Subset of Subaward Contracts</p>
			''', unsafe_allow_html = True)

		# Map everything to the page.
		folium_static(map, width = 725, height = 500)
# If ML model chosen is OPTICS.
elif model_type == "OPTICS":
	# Run function to get results.
	coords, data_final_test, distances, elbow_point_y = get_optimal(data_final, model_type, naics_code)
	
	# Collect the Latitude and Longitude of the contract award data.
	coords = coords[["Latitude", "Longitude"]]

	# Instantiate and fit the OPTICS model.
	optics = OPTICS(min_samples = 3, algorithm = "auto", metric = "haversine").fit(np.radians(coords))

	# Identify the unique cluster labels.
	cluster_labels = set(optics.labels_)

	# Remove the outlier from the cluster labels.
	cluster_labels.discard(-1)

	# Identify the number of clusters without outlier.
	n_clusters = len(cluster_labels)

	# Include header for section.
	st.markdown("#### 3. OPTICS Clustering Results")

	# Show how many clusters were identified.
	st.markdown(f"Number of Clusters (Without Outliers): {n_clusters}")

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
	
	# Add cluster label to dataframe.
	cluster_results["Cluster Label"] = \
	(cluster_results["Cluster"]
	 .astype(int)
	) + 1 

	# Convert cluster data into a Geopandas Data Frame and remove outlier cluster group (now 0).
	cluster_results_test = geopandas.GeoDataFrame(cluster_results, geometry = geopandas.points_from_xy(cluster_results.Longitude, cluster_results.Latitude)).query("Cluster != -1")

	# Add cluster data to world map.
	cluster_results_test.explore(
	    m = map, # Pass the map object.
	    tooltip = ["Cluster Label", "Cluster Average Subaward Amount", "Number of Subawardees in Cluster"], # Show custom tooltip.
	    color = "black" # Color the points.
	)

	# Include title to figure.
	st.markdown(r'''
	    <p style="text-align: center;"">Figure 2B. Map of OPTICS Clusters for Subset of Subaward Contracts</p>
	    ''', unsafe_allow_html = True)

	# Map everything to the page.
	folium_static(map, width = 725, height = 500)