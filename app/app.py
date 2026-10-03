"""
Retail Outlet Location Clustering - Interactive Web Application
Built with Streamlit and Plotly
B.Tech CSE (Semester V) Machine Learning Case Study

This interactive web tool loads the trained K-Means model, imputer, scaler, and PCA transformer.
It accepts candidate location parameters, validates inputs, predicts the location cluster,
computes distances to all cluster centroids, projects the candidate point onto the 2D PCA space,
and provides expansion suitability notes framed as strategic considerations.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Retail Location Clustering System",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a polished, modern interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .cluster-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load artifacts with caching
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    imputer = joblib.load(os.path.join(models_dir, "imputer.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    pca = joblib.load(os.path.join(models_dir, "pca_model.joblib"))

    with open(os.path.join(models_dir, "cluster_metadata.json"), "r") as f:
        metadata = json.load(f)

    df_data = pd.read_csv(os.path.join(data_dir, "retail_locations.csv"))
    # Pre-impute background dataset for PCA display
    X_imputed_bg = imputer.transform(df_data[metadata["feature_cols"]])
    X_scaled_bg = scaler.transform(X_imputed_bg)
    X_pca_bg = pca.transform(X_scaled_bg)
    df_bg = pd.DataFrame(X_pca_bg, columns=["PC1", "PC2"])
    df_bg["Cluster_ID"] = kmeans.predict(X_scaled_bg)
    df_bg["Cluster_Name"] = df_bg["Cluster_ID"].astype(str).map(metadata["cluster_mapping"])

    return imputer, scaler, kmeans, pca, metadata, df_bg

try:
    imputer, scaler, kmeans, pca, metadata, df_bg = load_ml_assets()
except Exception as e:
    st.error(f"Error loading models. Please ensure `scripts/train_and_evaluate.py` has been executed. Details: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">🏬 Retail Outlet Location Clustering System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">B.Tech CSE (Semester V) Machine Learning Case Study &mdash; Unsupervised Location Categorization & PCA Decision Space</div>', unsafe_allow_html=True)

# Sidebar: Candidate Location Input Controls
st.sidebar.header("📍 Candidate Location Parameters")
st.sidebar.markdown("Specify or adjust candidate location metrics below:")

# Form with defaults and range constraints
with st.sidebar.form("location_input_form"):
    pop_density = st.number_input(
        "Population Density (people/km²)",
        min_value=500.0,
        max_value=50000.0,
        value=14500.0,
        step=500.0,
        help="Local residential/commercial density per square kilometer."
    )
    
    avg_income = st.number_input(
        "Avg. Household Income ($/year)",
        min_value=15000.0,
        max_value=300000.0,
        value=85000.0,
        step=2500.0,
        help="Mean annual household income in the catchment zone."
    )
    
    competitors = st.slider(
        "Competitor Count (within 2 km)",
        min_value=0,
        max_value=40,
        value=12,
        help="Number of direct commercial competitors operating in proximity."
    )
    
    footfall = st.number_input(
        "Footfall Estimate (pedestrians/day)",
        min_value=500.0,
        max_value=100000.0,
        value=32000.0,
        step=1000.0,
        help="Estimated daily pedestrian transit passing the candidate storefront."
    )
    
    rental_cost = st.number_input(
        "Rental Cost ($/sq. ft./month)",
        min_value=5.0,
        max_value=300.0,
        value=85.0,
        step=5.0,
        help="Monthly commercial lease rate per square foot."
    )
    
    transport_score = st.slider(
        "Transport Accessibility Score",
        min_value=1.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
        help="1 = Remote/Isolated, 10 = Major Multimodal Transit Hub (Metro/Train/Bus/Highways)."
    )

    submit_btn = st.form_submit_button("🔍 Classify Location", use_container_width=True)

# Process candidate location data
input_df = pd.DataFrame([{
    "population_density": pop_density,
    "avg_household_income": avg_income,
    "competitor_count": competitors,
    "footfall_estimate": footfall,
    "rental_cost": rental_cost,
    "transport_accessibility": transport_score
}])

# Pipeline execution: Imputation (if NaN) -> StandardScaler -> K-Means Prediction
input_imputed = imputer.transform(input_df[metadata["feature_cols"]])
input_scaled = scaler.transform(input_imputed)

# Predict cluster
predicted_cluster_id = int(kmeans.predict(input_scaled)[0])
predicted_cluster_name = metadata["cluster_mapping"][str(predicted_cluster_id)]

# Centroid distances and proximity scores
cluster_centers_scaled = kmeans.cluster_centers_
distances = np.linalg.norm(cluster_centers_scaled - input_scaled, axis=1)

# Invert distances to compute softmax-like proximity percentage
inv_distances = 1.0 / (distances + 1e-6)
proximity_weights = (inv_distances / np.sum(inv_distances)) * 100.0

# 2D PCA projection of input
input_pca = pca.transform(input_scaled)
pca_centroids = pca.transform(cluster_centers_scaled)

# Main Dashboard Layout (2 Columns)
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("🎯 Classification Result")
    
    # Color badge
    badge_colors = {
        "Dense Premium City-Centre": ("#4F46E5", "#EEF2FF"),
        "Affluent Quiet Suburbs": ("#059669", "#ECFDF5"),
        "Busy Budget Markets": ("#D97706", "#FFFBEB"),
        "Low-Density Peripherals / Outskirts": ("#6B7280", "#F3F4F6"),
        "Transit-Hub Commercial Zones": ("#DC2626", "#FEF2F2")
    }
    fg_col, bg_col = badge_colors.get(predicted_cluster_name, ("#1E293B", "#F1F5F9"))

    st.markdown(
        f'<div style="background-color: {bg_col}; color: {fg_col}; border: 2px solid {fg_col};" class="cluster-badge">'
        f'Assigned Category: Cluster {predicted_cluster_id} &mdash; {predicted_cluster_name}'
        f'</div>',
        unsafe_allow_html=True
    )

    notes = metadata["suitability_notes"].get(predicted_cluster_name, {})
    
    st.markdown(f"**🏬 Target Business Profile:** {notes.get('business_persona', 'N/A')}")
    st.markdown(f"**✅ Strategic Advantages:** {notes.get('advantages', 'N/A')}")
    st.markdown(f"**⚠️ Key Expansion Considerations:** {notes.get('expansion_considerations', 'N/A')}")

    st.markdown(
        f'<div class="disclaimer-box">'
        f'<strong>📌 Expansion Note (Non-Ranking Principle):</strong> {notes.get("non_ranking_disclaimer", "")}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("📊 Cluster Proximity & Distance Breakdown")
    st.caption("Euclidean distance in standardized 6D feature space to each cluster center:")

    dist_data = []
    for c_id in range(metadata["optimal_k"]):
        c_name = metadata["cluster_mapping"][str(c_id)]
        dist_data.append({
            "Cluster ID": f"Cluster {c_id}",
            "Category Name": c_name,
            "Standardized Distance": round(distances[c_id], 3),
            "Affinity / Proximity (%)": f"{proximity_weights[c_id]:.1f}%"
        })
    df_dist = pd.DataFrame(dist_data)
    st.dataframe(df_dist, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🗺️ 2D PCA Cluster Map")
    st.caption(f"Projection into 2 Principal Components (Explaining {sum(metadata['pca_explained_variance']):.1%} total variance).")

    # Interactive Plotly PCA Plot
    fig = go.Figure()

    # Background points
    colors_map = {
        "Dense Premium City-Centre": "#6366F1",
        "Affluent Quiet Suburbs": "#10B981",
        "Busy Budget Markets": "#F59E0B",
        "Low-Density Peripherals / Outskirts": "#9CA3AF",
        "Transit-Hub Commercial Zones": "#EF4444"
    }

    for name, grp in df_bg.groupby("Cluster_Name"):
        fig.add_trace(go.Scatter(
            x=grp["PC1"],
            y=grp["PC2"],
            mode="markers",
            name=name,
            marker=dict(
                color=colors_map.get(name, "#888888"),
                size=7,
                opacity=0.45
            ),
            hoverinfo="text",
            hovertext=grp["Cluster_Name"]
        ))

    # Centroids
    fig.add_trace(go.Scatter(
        x=pca_centroids[:, 0],
        y=pca_centroids[:, 1],
        mode="markers+text",
        name="Cluster Centroids",
        text=[f"C{i}" for i in range(metadata["optimal_k"])],
        textposition="top center",
        textfont=dict(size=12, color="black"),
        marker=dict(
            symbol="x",
            size=14,
            color="black",
            line=dict(width=2, color="white")
        ),
        hoverinfo="text",
        hovertext=[f"Centroid {i}: {metadata['cluster_mapping'][str(i)]}" for i in range(metadata['optimal_k'])]
    ))

    # User Input Point
    fig.add_trace(go.Scatter(
        x=[input_pca[0, 0]],
        y=[input_pca[0, 1]],
        mode="markers+text",
        name="⭐ Your Candidate Location",
        text=["Candidate"],
        textposition="bottom center",
        textfont=dict(size=13, color="#DC2626", family="Arial Black"),
        marker=dict(
            symbol="star",
            size=22,
            color="#FF0000",
            line=dict(width=2, color="black")
        ),
        hoverinfo="text",
        hovertext=f"Your Location -> Predicted: {predicted_cluster_name}"
    ))

    fig.update_layout(
        xaxis_title=f"Principal Component 1 ({metadata['pca_explained_variance'][0]:.1%} variance)",
        yaxis_title=f"Principal Component 2 ({metadata['pca_explained_variance'][1]:.1%} variance)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
        plot_bgcolor="#FAFAFA"
    )

    st.plotly_chart(fig, use_container_width=True)

# Bottom Section: Comparison of Input vs Cluster Archetype Averages
st.markdown("---")
st.subheader("📈 Input Metrics vs. Cluster Archetype Benchmark")

benchmark_means = metadata["cluster_means"][str(predicted_cluster_id)]
comparison_rows = []
for col in metadata["feature_cols"]:
    inp_val = input_df[col].values[0]
    bench_val = benchmark_means[col]
    diff = inp_val - bench_val
    diff_pct = (diff / bench_val) * 100.0 if bench_val != 0 else 0.0
    comparison_rows.append({
        "Feature": metadata["feature_labels"][col],
        "Your Location Value": f"{inp_val:,.1f}",
        f"{predicted_cluster_name} Archetype Mean": f"{bench_val:,.1f}",
        "Deviation": f"{diff:+,.1f} ({diff_pct:+.1f}%)"
    })

df_comp = pd.DataFrame(comparison_rows)
st.dataframe(df_comp, use_container_width=True, hide_index=True)

# Technical Model Info
with st.expander("ℹ️ Model Architecture & Technical Deployment Details"):
    st.markdown(f"""
    - **Clustering Model**: `scikit-learn.cluster.KMeans` ($K = {metadata['optimal_k']}$, Initialization: `k-means++`, $n\\_init = 25$)
    - **Missing Value Handling**: `scikit-learn.impute.KNNImputer` ($k = 5$ nearest neighbors)
    - **Feature Preprocessing**: `scikit-learn.preprocessing.StandardScaler` (zero mean, unit variance scaling)
    - **Dimensionality Reduction**: `scikit-learn.decomposition.PCA` ($n = 2$, explained variance = {sum(metadata['pca_explained_variance']):.1%})
    - **Deployment Mode**: Live Inference via `joblib` deserialization
    - **Why K-Means is used for production inference instead of Hierarchical Agglomerative Clustering**: Hierarchical clustering computes pairwise similarity matrices between training instances without defining persistent parametric centroids or Voronoi decision boundaries. Consequently, Agglomerative clustering cannot classify unseen points $O(1)$ without re-computing the full $\\mathcal{{O}}(N^2)$ dendrogram from scratch. K-Means provides explicit cluster centroids $\\mu_k$, enabling instant Euclidean nearest-centroid classification for real-time applications.
    """)
