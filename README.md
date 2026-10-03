# Retail Outlet Location Clustering Using Machine Learning
**B.Tech Computer Science & Engineering (Semester V) — Machine Learning Case Study**

---

## 📌 Project Overview
This project presents an end-to-end unsupervised machine learning case study designed to segment candidate retail store locations across 6 socio-economic and spatial dimensions:
- `population_density` (people / km²)
- `avg_household_income` ($ / year)
- `competitor_count` (competing retail outlets within 2 km)
- `footfall_estimate` (daily pedestrian traffic)
- `rental_cost` ($ / sq. ft. / month)
- `transport_accessibility` (multimodal transit connectivity score: 1.0 – 10.0)

The project compares **$K$-Means Clustering** and **Hierarchical (Agglomerative) Clustering**, validates cluster counts using the **Elbow Method (WCSS)**, **Silhouette Score**, and **Ward Dendrogram**, validates cluster stability through **multi-seed and bootstrap resampling**, reduces dimensionality via **PCA (2D visualization)**, and deploys an interactive **Streamlit web application** for real-time location classification and expansion decision support.

---

## 🗂️ Project Structure

```text
Cross_platform_cycle_go_43_134/
├── data/
│   └── retail_locations.csv             # 800 synthetic location records with injected missing values
├── models/
│   ├── imputer.joblib                    # Trained KNNImputer (k=5)
│   ├── scaler.joblib                     # Trained StandardScaler
│   ├── kmeans_model.joblib               # Trained KMeans (K=5)
│   ├── pca_model.joblib                  # Fitted PCA (2 components, 90.4% variance explained)
│   └── cluster_metadata.json            # Category profiles, centroids, and suitability notes
├── figures/                              # High-resolution academic visualization exports
│   ├── eda_distributions.png            # Histograms and KDE plots
│   ├── eda_boxplots.png                 # Outlier boxplots
│   ├── correlation_heatmap.png          # Pearson correlation matrix
│   ├── elbow_silhouette_curve.png       # WCSS Elbow & Silhouette score vs. K
│   ├── dendrogram_cut.png               # Ward's linkage dendrogram with 5-cluster cut
│   ├── cluster_size_distribution.png    # K-Means vs. Agglomerative cluster balance
│   ├── stability_ari_distribution.png   # Multi-seed & Bootstrap ARI stability
│   ├── pca_cluster_scatter.png          # 2D PCA cluster space with centroids
│   └── cluster_profile_heatmap.png      # Standardized centroid Z-score heatmap
├── scripts/
│   ├── generate_data.py                  # Generates synthetic data with 5 urban archetypes
│   ├── train_and_evaluate.py             # Complete training, evaluation, and plotting script
│   └── create_notebook.py                # Notebook builder script
├── notebooks/
│   └── retail_location_clustering.ipynb # Clean, step-by-step commented Jupyter notebook
├── app/
│   └── app.py                            # Interactive Streamlit Web Application
├── report.md                             # Comprehensive project report with answers to all questions
├── viva_prep_sheet.md                    # 17 Viva Voce questions & model answers
├── requirements.txt                      # Project dependency specifications
└── README.md                             # Setup and execution guide
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Clone / Navigate to Project Directory
```bash
cd Cross_platform_cycle_go_43_134
```

### 3. Create & Activate Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # On macOS/Linux
# venv\Scripts\activate     # On Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Step 1: Generate Synthetic Dataset
To re-generate the synthetic dataset with 5 ground-truth archetypes and injected missing values:
```bash
python3 scripts/generate_data.py
```

### Step 2: Train Models & Generate All Figures
To run data imputation, scaling, $K$ evaluation, comparative modeling, PCA, and save all `.joblib` artifacts and figures:
```bash
python3 scripts/train_and_evaluate.py
```

### Step 3: Run the Jupyter Notebook
To explore the step-by-step interactive workflow in Jupyter:
```bash
jupyter notebook notebooks/retail_location_clustering.ipynb
```

### Step 4: Launch the Streamlit Web Application
To launch the interactive location classification dashboard:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏬 Identified Location Categories ($K = 5$)

| Cluster ID | Business Persona Name | Key Characteristics | Recommended Store Format |
| :---: | :--- | :--- | :--- |
| **0** | **Busy Budget Markets** | High density, lower income, heavy competition, high footfall, moderate rent. | Value fashion, discount grocery, quick-service food. |
| **1** | **Affluent Quiet Suburbs** | Low density, highest income, low competition, low footfall, moderate-high rent. | Specialty organic grocery, boutique wellness, luxury home decor. |
| **2** | **Dense Premium City-Centre** | High density, high income, high competition, high footfall, highest rent. | Flagship boutiques, experiential showrooms, luxury apparel. |
| **3** | **Low-Density Peripherals** | Lowest density, modest income, lowest competition, lowest footfall, lowest rent. | Big-box home improvement warehouses, wholesale discount stores. |
| **4** | **Transit-Hub Commercial Zones** | Moderate density, upper-middle income, moderate competition, peak footfall, high transit. | Convenience kiosks, express coffee, commuter pharmacies. |

---

## 🔑 Key Academic Insights
1. **Feature Scaling is Mandatory**: Distance-based clustering on unscaled data is completely distorted by variables with large numerical magnitudes (`avg_household_income`). `StandardScaler` ensures isotropic feature weight.
2. **Multi-Metric Validation ($K = 5$)**: The choice of $K=5$ is mathematically confirmed by the Elbow knee (WCSS drop from $986.3$ to $184.2$), peak Silhouette Score ($0.7814$), minimum Davies-Bouldin Index ($0.3125$), and cophenetic dendrogram branch heights.
3. **Partition Agreement ($\text{ARI} = 1.0000$)**: $K$-Means and Hierarchical Agglomerative clustering achieve complete agreement, confirming geometric cluster robustness.
4. **Why $K$-Means for Web Deployment**: Hierarchical clustering lacks parametric centroids and cannot classify new query points without an $\mathcal{O}(N^2)$ re-computation. $K$-Means stores centroids $\mu_k$, enabling instant $\mathcal{O}(K \cdot D)$ inference.
5. **Non-Ranking Principle**: Clustering groups locations by demographic and spatial similarity; **it does not rank locations by commercial attractiveness**. Attractiveness is contingent upon business model compatibility.

---

## 📚 Deliverables
- [Comprehensive Project Report](file:///Users/rohansamanta/Desktop/Cross_platform_cycle_go_43_134/report.md)
- [Viva Voce Exam Preparation Sheet](file:///Users/rohansamanta/Desktop/Cross_platform_cycle_go_43_134/viva_prep_sheet.md)
- [Jupyter Notebook](file:///Users/rohansamanta/Desktop/Cross_platform_cycle_go_43_134/notebooks/retail_location_clustering.ipynb)
- [Streamlit Web App Code](file:///Users/rohansamanta/Desktop/Cross_platform_cycle_go_43_134/app/app.py)
