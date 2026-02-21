import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
import shap

def load_dataset(base_dir):
    """
    Loads S1, Terrain, and Labels from the dataset directory.
    Returns: X (features), y (labels), feature_names
    """
    s1_dir = os.path.join(base_dir, "S1")
    terrain_dir = os.path.join(base_dir, "Terrain")
    label_dir = os.path.join(base_dir, "labels")
    
    if not os.path.exists(s1_dir):
        raise FileNotFoundError(f"Dataset directory {base_dir} not found or incomplete. Please run run_data_collection.py first.")

    tiles = [f for f in os.listdir(s1_dir) if f.endswith('.tif')]
    print(f"Found {len(tiles)} tiles.")
    
    X_list = []
    y_list = []
    
    for tile in tiles:
        # Load S1
        with rasterio.open(os.path.join(s1_dir, tile)) as src:
             s1 = src.read() # (3, H, W) -> VV, VH, Ratio
             s1 = np.moveaxis(s1, 0, -1) # (H, W, 3)
             
        # Load Terrain
        with rasterio.open(os.path.join(terrain_dir, tile)) as src:
             terrain = src.read() # (4, H, W) -> Elev, Slope, HAND, WaterSeasonality
             terrain = np.moveaxis(terrain, 0, -1) # (H, W, 4)
             
        # Load Label
        with rasterio.open(os.path.join(label_dir, tile)) as src:
             label = src.read(1) # (H, W)
             
        # Combine Features
        # S1 (3) + Terrain (4) = 7 features
        features = np.concatenate([s1, terrain], axis=2) # (H, W, 7)
        
        # Flatten
        features_flat = features.reshape(-1, 7)
        label_flat = label.reshape(-1)
        
        # Remove NoData or invalid pixels if necessary (e.g. padding)
        # Assuming all valid for now, or check for nan
        # S1 might have edge effects, but let's keep simple
        
        X_list.append(features_flat)
        y_list.append(label_flat)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Handle NaNs
    # Simple strategy: remove rows with NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    feature_names = [
        'S1_VV', 'S1_VH', 'S1_Ratio', 
        'Elevation', 'Slope', 'HAND', 'Water_Seasonality'
    ]
    
    print(f"Total Samples: {X.shape[0]}")
    print(f"Class Distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_names

def run_experiment(X_train, X_test, y_train, y_test, feature_indices, feature_names, experiment_name):
    """
    Trains RF on selected features and evaluates.
    """
    print(f"\n=== Running Experiment: {experiment_name} ===")
    selected_features = [feature_names[i] for i in feature_indices]
    print(f"Features: {selected_features}")
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train[:, feature_indices], y_train)
    
    y_pred = clf.predict(X_test[:, feature_indices])
    
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    return {
        'Experiment': experiment_name,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'Model': clf
    }

def analyze_feature_importance(model, X_sample, feature_names):
    """
    Analyzes feature importance using SHAP.
    """
    print("\n=== Feature Importance Analysis (SHAP) ===")
    
    # Tree Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values is a list [arrays for class 0, arrays for class 1]
    # We focus on class 1 (Flood)
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values
        
    # Summary Plot
    plt.figure()
    shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()
    print("SHAP Summary plot saved to shap_summary.png")
    
    # MDI Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nMean Decrease in Impurity (MDI) Importance:")
    for f in range(len(feature_names)):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

    # Plot MDI
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (MDI)")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance_mdi.png")
    plt.close()
    print("MDI Importance plot saved to feature_importance_mdi.png")

def main():
    dataset_dir = "dataset_sen12flood_v1"
    
    try:
        X, y, feature_names = load_dataset(dataset_dir)
    except Exception as e:
        print(e)
        return

    # Subsample for speed if dataset is huge
    if X.shape[0] > 100000:
        print("Subsampling data to 100k points for training...")
        idx = np.random.choice(X.shape[0], 100000, replace=False)
        X = X[idx]
        y = y[idx]
        
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define Experiments
    # S1 indices: 0,1,2
    # Elev: 3, Slope: 4, HAND: 5, Water: 6
    
    experiments = [
        ("SAR Only", [0, 1, 2]),
        ("SAR + HAND", [0, 1, 2, 5]),
        ("SAR + DEM", [0, 1, 2, 3]),
        ("SAR + HAND + Slope", [0, 1, 2, 5, 4]),
        ("SAR + Full Terrain", [0, 1, 2, 3, 4, 5, 6]),
    ]
    
    results = []
    
    for name, indices in experiments:
        res = run_experiment(X_train, X_test, y_train, y_test, indices, feature_names, name)
        results.append(res)
        
    # Validation Table
    df_res = pd.DataFrame(results)
    print("\n=== Ablation Results ===")
    print(df_res[['Experiment', 'F1', 'Precision', 'Recall']])
    df_res.to_csv("ablation_results.csv", index=False)
    
    # Feature Importance on Full Model
    full_model = results[-1]['Model']
    # Use a small sample for SHAP to be fast
    X_shap_sample = X_test[:1000, [0, 1, 2, 3, 4, 5, 6]]
    analyze_feature_importance(full_model, X_shap_sample, feature_names)

if __name__ == "__main__":
    main()
