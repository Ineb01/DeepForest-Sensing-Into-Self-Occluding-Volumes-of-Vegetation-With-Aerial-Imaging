import cv2
import numpy as np
import os
from pathlib import Path

# Directories to process
BASE_DIR = Path("../../data")
DATASET_FOLDERS = ["dataset_March", "dataset_May", "dataset_October"]
SUBFOLDERS = ["NDVI_renders/full", "NDVI_renders/cut"]
IMAGE_FILES = ["o.png", "x.png", "y.png", "z.png"]

# Rotation angle
ROTATION_ANGLE = -120

# This crops out the UI panels and keeps the central visualization area
CROP_REGION = (700, 300, 1650, 1340)

def rotate_image(image, angle):
    """Rotate image by specified angle with gray background"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Sample gray background color from corners
    gray_color = image[0, 0]
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image size to fit rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with gray background
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=tuple(gray_color.tolist()))
    
    return rotated

def crop_to_content(image, margin=5):
    """Crop image to tight bounding box around non-background pixels"""
    # Create mask of non-background pixels
    # Allow some tolerance for anti-aliasing
    tolerance = 0
    background_color = image[0, 0]
    diff = np.abs(image.astype(int) - background_color.astype(int))
    mask = np.any(diff > tolerance, axis=2).astype(np.uint8)
    
    # Find bounding box of non-background pixels
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add small margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        return image[y:y+h, x:x+w]
    
    return image

def process_images():
    """Process all images in all dataset folders"""
    print(f"Using fixed crop region: {CROP_REGION}")
    print(f"Rotation angle: {ROTATION_ANGLE}°\n")
    
    for folder in [f"{dataset}/{subfolder}" for dataset in DATASET_FOLDERS for subfolder in SUBFOLDERS]:
        folder_path = BASE_DIR / folder
        if not folder_path.exists():
            print(f"Skipping {folder} - directory not found")
            continue
        
        print(f"Processing {folder}...")
        
        for img_file in IMAGE_FILES:
            img_path = folder_path / img_file
            
            if not img_path.exists():
                print(f"  Skipping {img_file} - file not found")
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"  Error reading {img_file}")
                continue
            
            # Crop image using fixed region
            x1, y1, x2, y2 = CROP_REGION
            cropped = img[y1:y2, x1:x2]
            
            # Only rotate o.png, not the others
            if img_file == "o.png":
                rotated = rotate_image(cropped, ROTATION_ANGLE)
                processed = crop_to_content(rotated)
            else:
                processed = crop_to_content(cropped)
            
            # Save processed image
            output_path = folder_path / f"{img_file.replace('.png', '_processed.png')}"
            cv2.imwrite(str(output_path), processed)
            
            print(f"  Processed {img_file} -> {output_path.name}")
    
    print("\nAll images processed!")

if __name__ == "__main__":
    process_images()
