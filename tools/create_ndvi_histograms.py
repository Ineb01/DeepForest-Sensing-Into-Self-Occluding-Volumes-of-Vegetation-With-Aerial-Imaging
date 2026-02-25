import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path

# Base directory
BASE_DIR = Path("../../data")
DATASET_FOLDERS = ["dataset_March", "dataset_May", "dataset_October"]

# Numerical sort function
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_histogram_for_dataset(dataset_path, dataset_name):
    """Create histogram from all NDVI layers in a dataset"""
    ndvi_dir = dataset_path / "NDVI_layers"
    
    if not ndvi_dir.exists():
        print(f"  NDVI_layers directory not found for {dataset_name}")
        return
    
    # Load all NDVI layers
    all_ndvi_values = []
    
    npy_files = sorted(glob.glob(str(ndvi_dir / "*.npy")), key=numericalSort)
    
    if len(npy_files) == 0:
        print(f"  No .npy files found in {ndvi_dir}")
        return
    
    print(f"  Loading {len(npy_files)} NDVI layers...")
    
    for npy_file in npy_files:
        layer = np.load(npy_file)
        # Filter out NaN values and flatten
        valid_values = layer[~np.isnan(layer)].flatten()
        all_ndvi_values.extend(valid_values)
    
    all_ndvi_values = np.array(all_ndvi_values)
    
    print(f"  Before filtering: {len(all_ndvi_values)} values")
    print(f"  Values == -1: {np.sum(all_ndvi_values == -1)}")
    print(f"  Values == 0: {np.sum(all_ndvi_values == 0)}")
    print(f"  Values == 1: {np.sum(all_ndvi_values == 1)}")
    
    # Remove values that are exactly 0, 1, or -1
    all_ndvi_values = all_ndvi_values[(all_ndvi_values != 0) & 
                                       (all_ndvi_values != 1) & 
                                       (all_ndvi_values != -1)]
    
    print(f"  After filtering: {len(all_ndvi_values)} values")
    
    if len(all_ndvi_values) == 0:
        print(f"  No valid NDVI values found for {dataset_name}")
        return
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_ndvi_values, bins=100, color='green', alpha=0.7, edgecolor='black', range=(-1, 1))
    plt.xlabel('NDVI Value')
    plt.ylabel('Frequency')
    plt.title(f'NDVI Histogram - {dataset_name}')
    plt.xlim(-1, 1)  # Fixed x-axis scale for all histograms
    plt.ylim(0, 1e7)  # Fixed y-axis scale for all histograms
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(all_ndvi_values):.3f}\n'
    stats_text += f'Std: {np.std(all_ndvi_values):.3f}\n'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)
    
    # Save histogram
    output_path = dataset_path / "NDVI_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved histogram to {output_path.name}")
    print(f"  Statistics: Mean={np.mean(all_ndvi_values):.3f}, "
          f"Min={np.min(all_ndvi_values):.3f}, Max={np.max(all_ndvi_values):.3f}")

def create_all_histograms():
    """Create histograms for all datasets"""
    print("Creating NDVI histograms...\n")
    
    for folder in DATASET_FOLDERS:
        folder_path = BASE_DIR / folder
        
        if not folder_path.exists():
            print(f"Skipping {folder} - directory not found")
            continue
        
        print(f"Processing {folder}...")
        create_histogram_for_dataset(folder_path, folder)
        print()
    
    print("All histograms created!")

if __name__ == "__main__":
    create_all_histograms()
