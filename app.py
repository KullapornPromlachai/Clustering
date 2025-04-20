# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set the page config BEFORE any other Streamlit command
st.set_page_config(page_title="K-Means Clustering", layout="centered")

# Set title
st.title("K-Means Clustering Visualizer by Kullaporn Promlachai")

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load from a saved dataset or generate synthetic data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Display cluster centers
st.subheader("Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Data Points')
centroids = ax.scatter(loaded_model.cluster_centers_[:, 0],
                       loaded_model.cluster_centers_[:, 1],
                       s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
