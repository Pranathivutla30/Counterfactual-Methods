"""
Generate two comparison reports:
1. LightGCN results only (standard model)
2. Average across all 3 models (NCF, MF, LightGCN) per dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path(r"C:\Users\prana\OneDrive\Documents\thesis\pranathi3\results")

# Define datasets and models
datasets = ["yahoo", "coat", "kuairec"]
models = ["ncf", "mf", "lightgcn"]

# Load all CSV files
data = {}
for dataset in datasets:
    data[dataset] = {}
    for model in models:
        if model == "ncf":
            filename = f"debiasing_results_{dataset}.csv"
        else:
            filename = f"debiasing_results_{dataset}_{model}.csv"
        
        filepath = results_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[dataset][model] = df
            print(f"Loaded: {filename}")
        else:
            print(f"Missing: {filename}")

# Report 1: LightGCN Results Only
print("\n" + "="*80)
print("REPORT 1: LightGCN Results Only")
print("="*80)

lightgcn_report = []
for dataset in datasets:
    if "lightgcn" in data[dataset]:
        df = data[dataset]["lightgcn"]
        lightgcn_report.append(f"\n{'='*80}")
        lightgcn_report.append(f"Dataset: {dataset.upper()}")
        lightgcn_report.append(f"{'='*80}")
        lightgcn_report.append(df.to_string(index=False))
        lightgcn_report.append("")

lightgcn_text = "\n".join(lightgcn_report)

# Report 2: Average across all 3 models
print("\n" + "="*80)
print("REPORT 2: Average Across All Models (NCF, MF, LightGCN)")
print("="*80)

avg_report = []
for dataset in datasets:
    available_models = [m for m in models if m in data[dataset]]
    
    if not available_models:
        continue
    
    avg_report.append(f"\n{'='*80}")
    avg_report.append(f"Dataset: {dataset.upper()}")
    avg_report.append(f"Models included: {', '.join([m.upper() for m in available_models])}")
    avg_report.append(f"{'='*80}")
    
    # Get all metrics columns (excluding Model column)
    metric_cols = [col for col in data[dataset][available_models[0]].columns if col != "Model"]
    
    # Initialize average dataframe
    avg_df = data[dataset][available_models[0]][["Model"]].copy()
    
    # Calculate averages for each metric
    for col in metric_cols:
        values_list = []
        for model in available_models:
            values_list.append(data[dataset][model][col].values)
        
        # Stack all values and calculate mean
        stacked = np.array(values_list)
        avg_values = np.mean(stacked, axis=0)
        avg_df[col] = avg_values
    
    avg_report.append(avg_df.to_string(index=False))
    avg_report.append("")

avg_text = "\n".join(avg_report)

# Save both reports
output_dir = Path(r"C:\Users\prana\OneDrive\Documents\thesis\pranathi3\results")

# Report 1: LightGCN Only
with open(output_dir / "lightgcn_results_only.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("LIGHTGCN RESULTS ONLY (Standard Model)\n")
    f.write("="*80 + "\n")
    f.write(lightgcn_text)

# Report 2: Average Across Models
with open(output_dir / "average_across_all_models.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("AVERAGE RESULTS ACROSS ALL MODELS (NCF, MF, LightGCN)\n")
    f.write("="*80 + "\n")
    f.write(avg_text)

# Also create CSV versions for easier analysis
# LightGCN CSV
lightgcn_dfs = []
for dataset in datasets:
    if "lightgcn" in data[dataset]:
        df = data[dataset]["lightgcn"].copy()
        df.insert(0, "Dataset", dataset.upper())
        lightgcn_dfs.append(df)

if lightgcn_dfs:
    lightgcn_combined = pd.concat(lightgcn_dfs, ignore_index=True)
    lightgcn_combined.to_csv(output_dir / "lightgcn_results_only.csv", index=False)

# Average CSV
avg_dfs = []
for dataset in datasets:
    available_models = [m for m in models if m in data[dataset]]
    if not available_models:
        continue
    
    metric_cols = [col for col in data[dataset][available_models[0]].columns if col != "Model"]
    avg_df = data[dataset][available_models[0]][["Model"]].copy()
    
    for col in metric_cols:
        values_list = []
        for model in available_models:
            values_list.append(data[dataset][model][col].values)
        stacked = np.array(values_list)
        avg_values = np.mean(stacked, axis=0)
        avg_df[col] = avg_values
    
    avg_df.insert(0, "Dataset", dataset.upper())
    avg_dfs.append(avg_df)

if avg_dfs:
    avg_combined = pd.concat(avg_dfs, ignore_index=True)
    avg_combined.to_csv(output_dir / "average_across_all_models.csv", index=False)

print("\n" + "="*80)
print("REPORTS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("1. results/lightgcn_results_only.txt")
print("2. results/lightgcn_results_only.csv")
print("3. results/average_across_all_models.txt")
print("4. results/average_across_all_models.csv")
print("\n" + "="*80)

