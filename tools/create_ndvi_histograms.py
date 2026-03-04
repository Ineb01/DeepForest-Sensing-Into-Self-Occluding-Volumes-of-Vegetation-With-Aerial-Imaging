import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import vtk
from vtk.util import numpy_support

# Base directory
BASE_DIR = Path("../../data")
DATASET_FOLDERS = ["dataset_March", "dataset_May", "dataset_October"]

def load_ndvi_from_vti(vti_path):
    """Load NDVI values from a VTI file (channel 0 of 'Channels_and_opacity' scalars)"""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(vti_path))
    reader.Update()
    image_data = reader.GetOutput()

    scalars = image_data.GetPointData().GetScalars("Channels_and_opacity")
    if scalars is None:
        print(f"  'Channels_and_opacity' array not found in {vti_path.name}")
        return None

    # Shape: (440*440*440, 3) — channel 0 is NDVI
    data = numpy_support.vtk_to_numpy(scalars)
    ndvi_values = data[:, 0]
    return ndvi_values


def create_histogram_for_dataset(dataset_path, dataset_name, vti_filename, output_filename, title_suffix=""):
    """Create histogram from NDVI data stored in a VTI file"""
    vti_path = dataset_path / vti_filename

    if not vti_path.exists():
        print(f"  VTI file not found: {vti_filename}")
        return

    print(f"  Loading NDVI from {vti_filename}...")
    all_ndvi_values = load_ndvi_from_vti(vti_path)
    if all_ndvi_values is None:
        return

    # Drop NaN (masked/canopy voxels)
    all_ndvi_values = all_ndvi_values[~np.isnan(all_ndvi_values)]
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
    plt.title(f'NDVI Histogram - {dataset_name}{title_suffix}')
    plt.xlim(-1, 1)  # Fixed x-axis scale for all histograms
    plt.ylim(0, 5e6)  # Fixed y-axis scale for all histograms
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
    output_path = dataset_path / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved histogram to {output_path.name}")
    print(f"  Statistics: Mean={np.mean(all_ndvi_values):.3f}, "
          f"Min={np.min(all_ndvi_values):.3f}, Max={np.max(all_ndvi_values):.3f}")

def create_center_perspective_histogram(dataset_path, dataset_name):
    """Create NDVI histogram from center perspective RED and NIR reference images"""
    red_path = dataset_path / "RED_colmap_alignment" / "reference.jpg"
    nir_path = dataset_path / "NIR_colmap_alignment" / "reference.jpg"

    if not red_path.exists() or not nir_path.exists():
        print(f"  Reference images not found for {dataset_name}")
        return

    # Load as grayscale and convert to float
    red_img = np.array(Image.open(red_path).convert('L')).astype(float)
    nir_img = np.array(Image.open(nir_path).convert('L')).astype(float)

    # Resize to match if dimensions differ
    if red_img.shape != nir_img.shape:
        min_h = min(red_img.shape[0], nir_img.shape[0])
        min_w = min(red_img.shape[1], nir_img.shape[1])
        red_img = red_img[:min_h, :min_w]
        nir_img = nir_img[:min_h, :min_w]

    # Compute NDVI
    denominator = nir_img + red_img
    denominator[denominator == 0] = 1e-10
    ndvi = (nir_img - red_img) / denominator

    ndvi_values = ndvi.flatten()

    print(f"  Center perspective image shape: {red_img.shape}")
    print(f"  Total pixels: {len(ndvi_values)}")

    # Remove border pixels that are exactly 0 (black padding)
    mask = (red_img.flatten() > 0) | (nir_img.flatten() > 0)
    ndvi_values = ndvi_values[mask]

    print(f"  After removing black pixels: {len(ndvi_values)}")

    if len(ndvi_values) == 0:
        print(f"  No valid NDVI values found for {dataset_name}")
        return

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ndvi_values, bins=100, color='darkgreen', alpha=0.7, edgecolor='black', range=(-1, 1))
    plt.xlabel('NDVI Value')
    plt.ylabel('Frequency')
    plt.title(f'Center Perspective NDVI Histogram - {dataset_name}')
    plt.xlim(-1, 1)
    plt.ylim(0, 50000)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(ndvi_values):.3f}\n'
    stats_text += f'Std: {np.std(ndvi_values):.3f}\n'

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    # Save histogram
    output_path = dataset_path / "center_perspective_NDVI_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved histogram to {output_path.name}")
    print(f"  Statistics: Mean={np.mean(ndvi_values):.3f}, "
          f"Std={np.std(ndvi_values):.3f}, "
          f"Min={np.min(ndvi_values):.3f}, Max={np.max(ndvi_values):.3f}")


def create_all_histograms():
    """Create volumetric and center perspective NDVI histograms for all datasets"""
    print("Creating NDVI histograms...\n")

    for folder in DATASET_FOLDERS:
        folder_path = BASE_DIR / folder

        if not folder_path.exists():
            print(f"Skipping {folder} - directory not found")
            continue

        print(f"Processing {folder} (full canopy)...")
        create_histogram_for_dataset(
            folder_path, folder,
            vti_filename="corrected_NDVI_new.vti",
            output_filename="NDVI_histogram.png",
        )
        print()

        print(f"Processing {folder} (canopy removed)...")
        create_histogram_for_dataset(
            folder_path, folder,
            vti_filename="corrected_NDVI_removed_canopy.vti",
            output_filename="NDVI_histogram_no_canopy.png",
            title_suffix=" (canopy removed)",
        )
        print()

        print(f"Processing center perspective for {folder}...")
        create_center_perspective_histogram(folder_path, folder)
        print()

    print("All histograms created!")

if __name__ == "__main__":
    create_all_histograms()
