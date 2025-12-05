import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
import numpy as np
from PIL import Image
import glob
from tools.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# Simple slice contribution function to avoid matplotlib dependency
def slice_contribution_simple(x, y, z, total_layers, mode, start, use_random, axil):
    """Simplified version without matplotlib dependency"""
    # Load images from March NIR dataset
    nir_dir = '../data/dataset_March/NIR_layers_cropped'
    pattern = os.path.join(nir_dir, '*.png')
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    
    # Calculate slice range
    start_idx = max(0, z - axil//2)
    end_idx = min(len(file_list), z + axil//2 + 1)
    
    slices = []
    for i in range(start_idx, end_idx):
        if i < len(file_list):
            try:
                img = Image.open(file_list[i]).convert('L')
                img_array = np.array(img)
                slices.append(img_array)
            except Exception as e:
                print(f"Error loading {file_list[i]}: {e}")
                # Skip this image or use a placeholder
                continue
    
    return slices

def test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer):
    if layer <= 420:
        z = layer
        slices = slice_contribution_simple(x, y, z, 440, 'slices', 0, False, int(abs(z - 440) // 20))
    else:
        z = layer
        slices = slice_contribution_simple(x, y, z, 440, 'slices', 0, False, 1)
    
    v = [split_image_into_equal_tiles(value, 2) for value in slices]
    
    extra_layers = v[-1] if v else np.zeros((220, 220))
    if (abs(z - 440)) < 20:
        for i in range(20 - (abs(z - 440))):
            v.append(extra_layers)
    
    input_data = np.array(v)
    
    with torch.no_grad():
        inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
        test_outputs = model(inp.permute(1, 0, 2, 3, 4).float().to(device))
    
    output_value = test_outputs.item() * 255
    return (x, y, output_value)

def process_pixel(pixel_info, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer):
    # Load the model inside the process
    model = Simple3DCNN(in_channelss=model_axil)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.to(device)
    model.eval()
    
    x, y = pixel_info[0]
    return test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer)

def main(layer, model_axil):
    # Use March NIR cropped dataset
    ground_truth = '../data/dataset_March/NIR_layers_cropped'
    layer = int(layer)
    
    pattern = os.path.join(ground_truth, '*.png')
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    
    if layer - 1 >= len(file_list):
        print(f"Layer {layer} not found in dataset")
        return
    
    ground_truth_img = Image.open(file_list[layer - 1]).convert('L')
    print(f"Processing: {file_list[layer - 1]}, layer: {layer}")
    
    empty_image = Image.new('L', (440, 440), color=(0))
    random_matrix = np.zeros_like(empty_image)
    
    layer = 160 if layer > 420 else layer
    
    # Use correct checkpoint path
    model_save_path = f'../data/checkpoint/Layer_{layer}.pth'
    
    if not os.path.exists(model_save_path):
        print(f"Model checkpoint not found: {model_save_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    non_zero_pixels = []
    width, height = ground_truth_img.size
    
    # Collect all pixels (or just non-zero if preferred)
    for y in range(height):
        for x in range(width):
            pixel_value = ground_truth_img.getpixel((x, y))
            if True:  # Process all pixels
                non_zero_pixels.append(((x, y), pixel_value))
    
    print(f"Processing {len(non_zero_pixels)} pixels...")
    
    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_pixel, i, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer) for i in non_zero_pixels]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="pixel"):
            try:
                x, y, output_value = future.result()
                random_matrix[y, x] = output_value
            except Exception as exc:
                print(f'Error occurred: {exc}')
                # Continue processing other pixels
    
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the resulting image
    image = Image.fromarray(random_matrix, mode='L')
    output_path = os.path.join(output_dir, f"Layer_{layer}_march_nir.png")
    image.save(output_path)
    image.close()
    print(f"Saved result to: {output_path}")

if __name__ == '__main__':
    # Test with layer 1 from March NIR dataset
    layer = 1
    model_axil = 21  # From layers_data.txt for layer 1
    
    print(f"Testing layer {layer} with axil resolution {model_axil}")
    main(layer, model_axil)
