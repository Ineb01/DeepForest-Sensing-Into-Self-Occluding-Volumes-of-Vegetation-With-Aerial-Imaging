import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
import numpy as np
from PIL import Image
import glob
from tools.utils import *

def test_single_layer_generation(layer_num=1):
    """Test generation on a single 440x440 layer"""
    
    # Configuration
    input_dir = '../data/dataset_March/NIR_layers_cropped'
    output_dir = 'outputs/test_single_layer'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Testing Single Layer Generation ===")
    print(f"Layer: {layer_num}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Load layers_data.txt for model parameters
    with open('layers_data.txt', "r") as file:
        lines = file.readlines()
    
    model_axil = int(lines[layer_num - 1].split()[-1])
    print(f"Model axil resolution: {model_axil}")
    
    # Load input stack
    pattern = os.path.join(input_dir, '*.png')
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    
    print(f"Loading {len(file_list)} images...")
    input_stack = []
    for file_path in tqdm(file_list, desc="Loading"):
        img = Image.open(file_path).convert('L')
        input_stack.append(np.array(img))
        img.close()
    
    input_stack = np.array(input_stack)
    print(f"Input stack shape: {input_stack.shape}")
    
    # Verify dimensions
    if input_stack.shape[1:] != (440, 440):
        print(f"ERROR: Expected 440x440 images, got {input_stack.shape[1:]}!")
        return
    
    # Load model
    layer_for_model = 160 if layer_num > 420 else layer_num
    model_path = os.path.join('..', 'data', 'checkpoint', f'Layer_{layer_for_model}.pth')
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = Simple3DCNN(in_channelss=model_axil)
    device = torch.device('cpu')  # Use CPU since MPS doesn't support Conv3D
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Initialize output layer
    height, width = 440, 440
    output_layer = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate slice parameters
    if layer_num <= 420:
        z = layer_num
        axil_val = int(abs(z - 440) // 20)
    else:
        z = layer_num
        axil_val = 1
    
    print(f"Processing parameters: z={z}, axil_val={axil_val}")
    
    # Get slice range for this layer
    start_idx = max(0, (layer_num - 1) - axil_val//2)
    end_idx = min(input_stack.shape[0], (layer_num - 1) + axil_val//2 + 1)
    
    print(f"Using slices {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    
    # Extract and process slices
    layer_slices = input_stack[start_idx:end_idx]
    
    # Pad if necessary
    while layer_slices.shape[0] < axil_val:
        if layer_slices.shape[0] > 0:
            layer_slices = np.concatenate([layer_slices, layer_slices[-1:]], axis=0)
        else:
            print("ERROR: No slices available!")
            return
    
    if layer_slices.shape[0] > axil_val:
        layer_slices = layer_slices[:axil_val]
    
    print(f"Final slice count: {layer_slices.shape[0]} (expected: {axil_val})")
    
    # Pre-process slices into tiles
    print("Pre-processing tiles...")
    processed_slices = []
    for slice_idx in range(layer_slices.shape[0]):
        tiles = split_image_into_equal_tiles(layer_slices[slice_idx], 2)
        processed_slices.append(tiles)
    
    processed_slices = np.array(processed_slices)
    
    # Add extra layers if needed (matching original logic)
    if processed_slices.shape[0] > 0:
        extra_layers = processed_slices[-1:]
        if (abs(z - 440)) < 20:
            repeats = 20 - (abs(z - 440))
            extra_layers_repeated = np.repeat(extra_layers, repeats, axis=0)
            processed_slices = np.concatenate([processed_slices, extra_layers_repeated], axis=0)
    
    print(f"Processed slices shape: {processed_slices.shape}")
    
    # Process each pixel
    total_pixels = height * width
    print(f"Processing {total_pixels:,} pixels...")
    
    # Debug: Track output values
    output_values = []
    error_count = 0
    
    with tqdm(total=total_pixels, desc=f"Layer {layer_num}") as pbar:
        for y in range(height):
            for x in range(width):
                try:
                    # Use the pre-processed tiles
                    input_data = processed_slices
                    
                    with torch.no_grad():
                        inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
                        output = model(inp.permute(1, 0, 2, 3, 4).float().to(device))
                        raw_output = output.item()
                        output_value = raw_output * 255
                        
                        # Debug: Collect values for first 1000 pixels
                        if len(output_values) < 1000:
                            output_values.append((raw_output, output_value))
                        
                        output_layer[y, x] = int(np.clip(output_value, 0, 255))
                
                except Exception as e:
                    error_count += 1
                    # Keep original on error
                    if layer_num - 1 < input_stack.shape[0]:
                        output_layer[y, x] = input_stack[layer_num - 1, y, x]
                    
                    # Debug: Show first few errors
                    if error_count <= 5:
                        print(f"\nError at ({x},{y}): {e}")
                
                pbar.update(1)
    
    # Debug: Show output value statistics
    if output_values:
        raw_vals = [v[0] for v in output_values[:100]]  # First 100 raw values
        scaled_vals = [v[1] for v in output_values[:100]]  # First 100 scaled values
        print(f"\nDEBUG - First 100 model outputs:")
        print(f"  Raw values range: {min(raw_vals):.6f} to {max(raw_vals):.6f}")
        print(f"  Scaled values range: {min(scaled_vals):.1f} to {max(scaled_vals):.1f}")
        print(f"  Unique raw values: {len(set(raw_vals))}")
        print(f"  Error count: {error_count}")
        
        # Show first few actual values
        print(f"  Sample raw outputs: {raw_vals[:10]}")
    else:
        print(f"\nDEBUG - No output values collected! All {error_count} pixels failed.")
    
    # Save results
    print("Saving results...")
    
    # Save output layer
    output_image = Image.fromarray(output_layer, mode='L')
    output_path = os.path.join(output_dir, f"clean_layer_{layer_num:03d}.png")
    output_image.save(output_path)
    output_image.close()
    
    # Save original for comparison
    original_image = Image.fromarray(input_stack[layer_num - 1], mode='L')
    original_path = os.path.join(output_dir, f"original_layer_{layer_num:03d}.png")
    original_image.save(original_path)
    original_image.close()
    
    # Calculate some statistics
    original_stats = {
        'mean': input_stack[layer_num - 1].mean(),
        'std': input_stack[layer_num - 1].std(),
        'min': input_stack[layer_num - 1].min(),
        'max': input_stack[layer_num - 1].max()
    }
    
    output_stats = {
        'mean': output_layer.mean(),
        'std': output_layer.std(),
        'min': output_layer.min(),
        'max': output_layer.max()
    }
    
    print(f"  Original - Mean: {original_stats['mean']:.1f}, Std: {original_stats['std']:.1f}, Range: {original_stats['min']}-{original_stats['max']}")
    print(f"  Clean    - Mean: {output_stats['mean']:.1f}, Std: {output_stats['std']:.1f}, Range: {output_stats['min']}-{output_stats['max']}")
    return True

if __name__ == '__main__':
    # Test layer 300 (middle layer, should have more variation)
    success = test_single_layer_generation(layer_num=300)