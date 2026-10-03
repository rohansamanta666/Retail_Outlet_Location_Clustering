"""
Retail Outlet Location Clustering - Synthetic Dataset Generator
Author: B.Tech CSE (Semester V) Machine Learning Case Study

This script synthesizes 800 realistic candidate retail outlet locations across 6 features:
1. population_density (people/km^2)
2. avg_household_income (annual USD / currency units)
3. competitor_count (competing retail outlets within 2km)
4. footfall_estimate (average daily pedestrian traffic)
5. rental_cost (monthly rental cost in USD per sq. ft.)
6. transport_accessibility (index score from 1.0 to 10.0 based on transit nodes)

5 distinct urban commercial archetypes are synthesized with controlled variance and covariances.
6-8% missing values (NaN) are deliberately injected into footfall_estimate and rental_cost
to replicate real-world commercial surveying gaps.
"""

import os
import numpy as np
import pandas as pd

def generate_retail_dataset(
    n_samples: int = 800,
    random_seed: int = 42,
    missing_rate_footfall: float = 0.07,
    missing_rate_rent: float = 0.06,
    output_path: str = "data/retail_locations.csv"
) -> pd.DataFrame:
    np.random.seed(random_seed)

    # Define 5 archetypal cluster configurations:
    # Each archetype has (sample_fraction, feature_means, feature_std_devs)
    archetypes = [
        {
            "name": "Dense Premium City-Centre",
            "fraction": 0.22,
            "means": [18500, 115000, 19, 36000, 125, 9.1],
            "stds":  [2200,   12000,  3,  4500,  15, 0.4]
        },
        {
            "name": "Affluent Quiet Suburbs",
            "fraction": 0.20,
            "means": [4200,  148000,  4,  6500,  78, 5.4],
            "stds":  [800,   15000,  1.5, 1200,  10, 0.8]
        },
        {
            "name": "Busy Budget Markets",
            "fraction": 0.23,
            "means": [23000,  43000, 24, 44000,  48, 8.6],
            "stds":  [2800,    6000,  4,  5500,   8, 0.6]
        },
        {
            "name": "Low-Density Peripherals",
            "fraction": 0.17,
            "means": [2800,   49000,  3,  4000,  22, 3.2],
            "stds":  [600,    7000,  1.2,  900,   4, 0.7]
        },
        {
            "name": "Transit-Hub Commercial Zones",
            "fraction": 0.18,
            "means": [12500,  88000, 15, 54000,  98, 9.7],
            "stds":  [1800,   10000,  3,  6000,  12, 0.25]
        }
    ]

    features = [
        "population_density",
        "avg_household_income",
        "competitor_count",
        "footfall_estimate",
        "rental_cost",
        "transport_accessibility"
    ]

    data_rows = []
    ground_truth_labels = []

    for arch in archetypes:
        n_arch = int(n_samples * arch["fraction"])
        means = np.array(arch["means"])
        stds = np.array(arch["stds"])

        # Generate Gaussian samples with realistic feature bounds
        samples = np.random.normal(loc=means, scale=stds, size=(n_arch, len(features)))

        # Bounds and rounding constraints to maintain domain realism
        samples[:, 0] = np.clip(np.round(samples[:, 0]), 500, 40000)      # pop density
        samples[:, 1] = np.clip(np.round(samples[:, 1], -2), 15000, 250000) # income
        samples[:, 2] = np.clip(np.round(samples[:, 2]), 0, 40)          # competitor count
        samples[:, 3] = np.clip(np.round(samples[:, 3], -1), 500, 80000)   # footfall
        samples[:, 4] = np.clip(np.round(samples[:, 4], 2), 5.0, 250.0)   # rental cost
        samples[:, 5] = np.clip(np.round(samples[:, 5], 1), 1.0, 10.0)    # transport score

        data_rows.append(samples)
        ground_truth_labels.extend([arch["name"]] * n_arch)

    # Combine all archetypes
    all_data = np.vstack(data_rows)
    actual_n = len(all_data)

    df = pd.DataFrame(all_data, columns=features)
    df["location_id"] = [f"LOC_{i+1:04d}" for i in range(actual_n)]
    df["true_archetype"] = ground_truth_labels

    # Shuffle dataset rows to avoid sequential ordering
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Deliberately inject Missing Values (NaN) into footfall_estimate & rental_cost (MCAR)
    n_missing_ff = int(actual_n * missing_rate_footfall)
    n_missing_rent = int(actual_n * missing_rate_rent)

    missing_ff_idx = np.random.choice(df.index, size=n_missing_ff, replace=False)
    missing_rent_idx = np.random.choice(df.index, size=n_missing_rent, replace=False)

    df.loc[missing_ff_idx, "footfall_estimate"] = np.nan
    df.loc[missing_rent_idx, "rental_cost"] = np.nan

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated synthetic retail location dataset successfully at: {output_path}")
    print(f"Total candidate locations: {len(df)}")
    print(f"Missing values injected:")
    print(f" - footfall_estimate: {df['footfall_estimate'].isna().sum()} ({df['footfall_estimate'].isna().mean():.1%})")
    print(f" - rental_cost: {df['rental_cost'].isna().sum()} ({df['rental_cost'].isna().mean():.1%})")
    print("\nFeature Summary (Non-Null Values):")
    print(df[features].describe().T[["mean", "std", "min", "50%", "max"]])

    return df

if __name__ == "__main__":
    generate_retail_dataset()
