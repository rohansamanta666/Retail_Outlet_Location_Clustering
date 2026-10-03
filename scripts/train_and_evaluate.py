"""
Retail Outlet Location Clustering - Complete Pipeline & Comparative Evaluation
Author: B.Tech CSE (Semester V) Machine Learning Case Study

This script executes:
1. Data loading and missing value analysis
2. Imputation (Median / KNN) and Feature Scaling (StandardScaler)
3. EDA plot generation (distributions, boxplots, correlation heatmap)
4. Evaluation of optimal K (Elbow curve WCSS, Silhouette score, Davies-Bouldin index)
5. Hierarchical clustering dendrogram with Ward linkage and cut threshold
6. Side-by-side comparative analysis: K-Means vs Agglomerative Clustering (ARI agreement, cross-tabulation)
7. Stability testing via bootstrap resampling and multi-seed perturbation
8. PCA 2D projection and explained variance analysis
9. Cluster profiling, business persona tagging, and suitability considerations
10. Model persistence with joblib and cluster metadata JSON export
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Configure matplotlib styling for crisp, professional academic figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300
})

FEATURE_COLS = [
    "population_density",
    "avg_household_income",
    "competitor_count",
    "footfall_estimate",
    "rental_cost",
    "transport_accessibility"
]

FEATURE_LABELS = {
    "population_density": "Pop Density (people/km²)",
    "avg_household_income": "Avg Income ($/yr)",
    "competitor_count": "Competitor Count",
    "footfall_estimate": "Footfall (pedestrians/day)",
    "rental_cost": "Rental Cost ($/sq ft/mo)",
    "transport_accessibility": "Transport Score (1-10)"
}

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    data_path = "data/retail_locations.csv"
    if not os.path.exists(data_path):
        from generate_data import generate_retail_dataset
        generate_retail_dataset(output_path=data_path)

    df_raw = pd.read_csv(data_path)
    print(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # ==========================================
    # 1. Exploratory Data Analysis & Visualizations
    # ==========================================
    print("\n--- Generating EDA Visualizations ---")
    
    # Distributions & KDE
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, col in enumerate(FEATURE_COLS):
        sns.histplot(df_raw[col].dropna(), kde=True, ax=axes[i], color=colors[i], bins=25, edgecolor="black", alpha=0.6)
        axes[i].set_title(f"Distribution of {FEATURE_LABELS[col]}")
        axes[i].set_xlabel(FEATURE_LABELS[col])
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/eda_distributions.png")
    plt.close()

    # Boxplots for Outlier Inspection
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.boxplot(y=df_raw[col], ax=axes[i], color=colors[i], width=0.4, fliersize=4)
        axes[i].set_title(f"Boxplot: {FEATURE_LABELS[col]}")
        axes[i].set_ylabel(FEATURE_LABELS[col])
    plt.tight_layout()
    plt.savefig("figures/eda_boxplots.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr = df_raw[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                xticklabels=[FEATURE_LABELS[c] for c in FEATURE_COLS],
                yticklabels=[FEATURE_LABELS[c] for c in FEATURE_COLS],
                cbar_kws={"label": "Pearson Correlation"})
    plt.title("Feature Correlation Matrix (Raw Data)", pad=15)
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png")
    plt.close()

    # ==========================================
    # 2. Preprocessing: Imputation & Scaling
    # ==========================================
    print("\n--- Preprocessing (Imputation & Scaling) ---")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(df_raw[FEATURE_COLS])
    df_imputed = pd.DataFrame(X_imputed, columns=FEATURE_COLS)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    # ==========================================
    # 3. Determining Optimal Number of Clusters (K)
    # ==========================================
    print("\n--- Evaluating Optimal K (Elbow & Silhouette) ---")
    k_range = list(range(2, 11))
    wcss = []
    silhouette_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
        db_scores.append(davies_bouldin_score(X_scaled, km.labels_))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_wcss = "#1f77b4"
    ax1.set_xlabel("Number of Clusters (K)", fontweight="bold")
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", color=color_wcss, fontweight="bold")
    line1 = ax1.plot(k_range, wcss, "o-", color=color_wcss, linewidth=2.5, markersize=8, label="WCSS (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color_wcss)
    ax1.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="Selected K = 5")

    ax2 = ax1.twinx()
    color_sil = "#d62728"
    ax2.set_ylabel("Silhouette Score", color=color_sil, fontweight="bold")
    line2 = ax2.plot(k_range, silhouette_scores, "s--", color=color_sil, linewidth=2.5, markersize=8, label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)

    # Combined legend
    lines = line1 + line2 + [plt.Line2D([0], [0], color="gray", linestyle="--", label="Selected K=5")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title("Optimal K Evaluation: Elbow Curve and Silhouette Analysis", pad=15)
    plt.tight_layout()
    plt.savefig("figures/elbow_silhouette_curve.png")
    plt.close()

    # Hierarchical Clustering Dendrogram
    print("\n--- Generating Hierarchical Dendrogram ---")
    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(14, 7))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
    # 5 clusters cut line
    # Find distance threshold that gives 5 clusters
    last_merges = Z[-(5-1):, 2]
    cut_threshold = (last_merges[0] + Z[-(5), 2]) / 2.0
    plt.axhline(y=cut_threshold, color="crimson", linestyle="--", linewidth=2, label=f"5-Cluster Cut Threshold (Distance ≈ {cut_threshold:.1f})")
    plt.title("Hierarchical Agglomerative Dendrogram (Ward's Linkage)", pad=15)
    plt.xlabel("Sample Index or (Cluster Size)")
    plt.ylabel("Cophenetic Distance / Ward Dissimilarity")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/dendrogram_cut.png")
    plt.close()

    # ==========================================
    # 4. Model Training: K-Means & Agglomerative
    # ==========================================
    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=25, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    hier_labels = hierarchical.fit_predict(X_scaled)

    # ARI Agreement
    ari = adjusted_rand_score(kmeans_labels, hier_labels)
    print(f"\nAdjusted Rand Index (K-Means vs Hierarchical): {ari:.4f}")

    # Cluster Size Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    km_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    hier_counts = pd.Series(hier_labels).value_counts().sort_index()

    sns.barplot(x=km_counts.index, y=km_counts.values, ax=ax1, palette="Blues_d")
    ax1.set_title("K-Means Cluster Size Distribution (K=5)")
    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Number of Candidate Locations")
    for i, v in enumerate(km_counts.values):
        ax1.text(i, v + 3, str(v), ha="center", fontweight="bold")

    sns.barplot(x=hier_counts.index, y=hier_counts.values, ax=ax2, palette="Oranges_d")
    ax2.set_title("Hierarchical (Agglomerative) Cluster Size Distribution (K=5)")
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Number of Candidate Locations")
    for i, v in enumerate(hier_counts.values):
        ax2.text(i, v + 3, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/cluster_size_distribution.png")
    plt.close()

    # ==========================================
    # 5. Stability Analysis (Multi-Seed & Resampling)
    # ==========================================
    print("\n--- Performing Cluster Stability Analysis ---")
    seed_aris = []
    base_labels = kmeans_labels
    for seed in range(1, 51):
        km_test = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=seed)
        labels_test = km_test.fit_predict(X_scaled)
        seed_aris.append(adjusted_rand_score(base_labels, labels_test))

    # Bootstrap Resampling (100 iterations on 85% sample)
    bootstrap_aris = []
    n_pts = len(X_scaled)
    np.random.seed(42)
    for _ in range(100):
        sample_idx = np.random.choice(n_pts, size=int(0.85 * n_pts), replace=False)
        km_boot = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)
        boot_labels = km_boot.fit_predict(X_scaled[sample_idx])
        base_sub_labels = base_labels[sample_idx]
        bootstrap_aris.append(adjusted_rand_score(base_sub_labels, boot_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(seed_aris, bins=15, kde=True, ax=ax1, color="#2ca02c", edgecolor="black")
    ax1.set_title(f"Multi-Seed ARI Stability (Mean = {np.mean(seed_aris):.4f})")
    ax1.set_xlabel("Adjusted Rand Index (ARI)")
    ax1.set_ylabel("Count")

    sns.histplot(bootstrap_aris, bins=15, kde=True, ax=ax2, color="#9467bd", edgecolor="black")
    ax2.set_title(f"Bootstrap Subsample ARI Stability (Mean = {np.mean(bootstrap_aris):.4f})")
    ax2.set_xlabel("Adjusted Rand Index (ARI)")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("figures/stability_ari_distribution.png")
    plt.close()

    # ==========================================
    # 6. PCA 2D Visualization
    # ==========================================
    print("\n--- Fitting 2D PCA & Plotting ---")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_centroids = pca.transform(kmeans.cluster_centers_)

    exp_var = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1={exp_var[0]:.1%}, PC2={exp_var[1]:.1%}, Total={sum(exp_var):.1%}")

    # Map clusters to descriptive business archetypes based on profile characteristics
    # Compute unscaled cluster means to reliably identify cluster personas
    df_imputed["kmeans_cluster"] = kmeans_labels
    cluster_means = df_imputed.groupby("kmeans_cluster")[FEATURE_COLS].mean()

    # Match each cluster ID to its closest archetype by logic:
    # 1. Highest footfall & transport -> Transit-Hub
    # 2. Highest income & low footfall -> Affluent Quiet Suburbs
    # 3. Highest population density & high competitor & low income -> Busy Budget Markets
    # 4. Highest rent & high density & high income -> Dense Premium City-Centre
    # 5. Lowest density & lowest rent -> Low-Density Peripherals
    
    cluster_mapping = {}
    for c_id in range(optimal_k):
        m = cluster_means.loc[c_id]
        if m["avg_household_income"] > 130000:
            cluster_mapping[c_id] = "Affluent Quiet Suburbs"
        elif m["footfall_estimate"] > 48000 and m["transport_accessibility"] > 9.0:
            cluster_mapping[c_id] = "Transit-Hub Commercial Zones"
        elif m["population_density"] > 18000 and m["avg_household_income"] < 55000:
            cluster_mapping[c_id] = "Busy Budget Markets"
        elif m["rental_cost"] > 100 or m["avg_household_income"] > 100000:
            cluster_mapping[c_id] = "Dense Premium City-Centre"
        else:
            cluster_mapping[c_id] = "Low-Density Peripherals / Outskirts"

    df_imputed["cluster_name"] = df_imputed["kmeans_cluster"].map(cluster_mapping)

    plt.figure(figsize=(11, 8))
    palette = sns.color_palette("tab10", optimal_k)
    for c_id in range(optimal_k):
        mask = (kmeans_labels == c_id)
        name = cluster_mapping[c_id]
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c_id}: {name}",
                    alpha=0.65, edgecolors="none", s=50, color=palette[c_id])

    # Plot Centroids
    plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=220, c="black", marker="X",
                edgecolors="white", linewidth=1.5, label="Cluster Centroids", zorder=10)

    for c_id in range(optimal_k):
        plt.annotate(f"C{c_id}", (pca_centroids[c_id, 0], pca_centroids[c_id, 1]),
                     xytext=(7, 7), textcoords="offset points", fontweight="bold", fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8, ec="black"))

    plt.title(f"2D PCA Projection of Retail Clusters (Explained Variance: {sum(exp_var):.1%})", pad=15)
    plt.xlabel(f"Principal Component 1 ({exp_var[0]:.1%} variance explained)")
    plt.ylabel(f"Principal Component 2 ({exp_var[1]:.1%} variance explained)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("figures/pca_cluster_scatter.png")
    plt.close()

    # ==========================================
    # 7. Cluster Profiling Visualizations (Heatmap & Radar)
    # ==========================================
    print("\n--- Generating Cluster Profiles & Visualizations ---")
    # Standardized profile heatmap for comparison
    scaled_centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=[FEATURE_LABELS[c] for c in FEATURE_COLS])
    scaled_centroids_df.index = [f"Cluster {i}: {cluster_mapping[i]}" for i in range(optimal_k)]

    plt.figure(figsize=(11, 6))
    sns.heatmap(scaled_centroids_df, annot=True, fmt=".2f", cmap="vlag", center=0, cbar_kws={"label": "Standardized Feature Deviation (Z-score)"})
    plt.title("Cluster Centroid Profiles (Z-Score Heatmap)", pad=15)
    plt.tight_layout()
    plt.savefig("figures/cluster_profile_heatmap.png")
    plt.close()

    # Business Expansion Considerations Dictionary
    suitability_notes = {
        "Dense Premium City-Centre": {
            "business_persona": "Flagship stores, premium lifestyle apparel, experiential luxury showrooms, specialty food & beverage.",
            "advantages": "High disposable income, dense foot traffic, exceptional public transit connectivity, prestige brand visibility.",
            "expansion_considerations": "Substantial rental overhead, intense competitor concentration, restricted store footprint, high customer acquisition costs.",
            "non_ranking_disclaimer": "Highly attractive for premium branding and margin-heavy categories, but unfavorable for low-margin discount retail."
        },
        "Affluent Quiet Suburbs": {
            "business_persona": "Boutique organic groceries, family healthcare clinics, premium fitness centers, home decor, high-end children's goods.",
            "advantages": "Affluent residential clientele, lower direct competitor density, high average order values, strong customer loyalty.",
            "expansion_considerations": "Low daily pedestrian footfall requiring targeted digital marketing, high dependency on personal vehicular transit, moderate rental rates.",
            "non_ranking_disclaimer": "Ideal for appointment-based or weekly destination shopping, but unsuitable for impulse-driven impulse retail."
        },
        "Busy Budget Markets": {
            "business_persona": "Value fashion, discount grocery supermarkets, quick-service street food, pharmacy chains, consumer electronics accessories.",
            "advantages": "Massive daily pedestrian volume, strong transit links, moderate rental rates relative to turnover volume.",
            "expansion_considerations": "High competitor saturation, price-sensitive customer base demanding aggressive discounting, lower basket sizes.",
            "non_ranking_disclaimer": "Outstanding for high-velocity volume retail, but unsuitable for luxury or high-margin positioning."
        },
        "Low-Density Peripherals / Outskirts": {
            "business_persona": "Big-box home improvement warehouses, wholesale discount outlets, automotive showrooms, bulk distribution centers.",
            "advantages": "Lowest rental costs per square foot, minimal direct competition, extensive physical space for parking and large floorplans.",
            "expansion_considerations": "Low organic footfall, low population density, weak public transit access necessitating dedicated highway marketing.",
            "non_ranking_disclaimer": "Optimal for footprint-intensive destination retail, but unviable for small convenience shops."
        },
        "Transit-Hub Commercial Zones": {
            "business_persona": "Grab-and-go convenience retail, quick-service dining, express coffee kiosks, travel essentials, commuter pharmacies.",
            "advantages": "Peak daily foot traffic, maximum multimodal transit connectivity, steady recurring commuter flow across all peak hours.",
            "expansion_considerations": "High rental costs per square foot, compact store footprints, rapid customer transit times requiring swift point-of-sale operations.",
            "non_ranking_disclaimer": "Superior for fast convenience transactions, but ill-suited for leisurely browse-and-linger retail concepts."
        }
    }

    # Save models and metadata
    import joblib
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(kmeans, "models/kmeans_model.joblib")
    joblib.dump(pca, "models/pca_model.joblib")

    # Build cluster metadata JSON
    metadata = {
        "optimal_k": optimal_k,
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "pca_explained_variance": [float(v) for v in exp_var],
        "cluster_mapping": {str(k): v for k, v in cluster_mapping.items()},
        "cluster_means": {
            str(k): {col: float(cluster_means.loc[k, col]) for col in FEATURE_COLS}
            for k in range(optimal_k)
        },
        "suitability_notes": suitability_notes
    }

    with open("models/cluster_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved all models, figures, and metadata successfully.")
    print("Files created in models/:")
    for f in os.listdir("models"):
        print(f" - models/{f}")

if __name__ == "__main__":
    main()
