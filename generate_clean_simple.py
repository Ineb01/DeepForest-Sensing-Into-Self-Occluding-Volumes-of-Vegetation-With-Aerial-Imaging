import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
import numpy as np
from PIL import Image
import glob
from tools.utils import *

def load_input_stack(input_dir):
    """Load the entire 440x440x440 input stack"""
    pattern = os.path.join(input_dir, '*.png')
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    
    print(f"Loading {len(file_list)} images...")
    images = []
    
    for file_path in tqdm(file_list, desc="Loading images"):
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        images.append(img_array)
        img.close()
    
    stack = np.array(images)
    print(f"Loaded stack shape: {stack.shape}")
    return stack

def slice_contribution_from_stack(stack, x, y, z, axil):
    """Extract slices around z-position for given x,y coordinate"""
    start_idx = max(0, z - axil//2)
    end_idx = min(stack.shape[0], z + axil//2 + 1)
    
    slices = []
    for i in range(start_idx, end_idx):
        slices.append(stack[i])
    
    # Pad to get exactly axil slices
    while len(slices) < axil:
        if slices:
            slices.append(slices[-1])
        else:
            break
    
    return slices

def process_single_layer(layer_idx, input_stack, model_axil):
    """Process a single layer (z-slice) of the stack"""
    
    layer = layer_idx + 1  # Convert to 1-based
    height, width = input_stack.shape[1], input_stack.shape[2]
    
    # Load model
    model_save_path = os.path.join('..', 'data', 'checkpoint', f'Layer_{layer if layer <= 420 else 160}.pth')
    
    if not os.path.exists(model_save_path):
        print(f"Model checkpoint not found: {model_save_path}")
        return None
    
    model = Simple3DCNN(in_channelss=model_axil)
    device = torch.device('cpu')  # Use CPU since MPS doesn't support Conv3D
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Processing layer {layer} ({height}x{width} pixels)...")
    
    # Initialize output layer
    output_layer = np.zeros((height, width), dtype=np.uint8)
    
    # Process each pixel
    total_pixels = height * width
    with tqdm(total=total_pixels, desc=f"Layer {layer}") as pbar:
        for y in range(height):
            for x in range(width):
                try:
                    # Get slices for this pixel
                    if layer <= 420:
                        z = layer
                        axil_val = int(abs(z - 440) // 20)
                    else:
                        z = layer
                        axil_val = 1
                    
                    slices = slice_contribution_from_stack(input_stack, x, y, z-1, axil_val)  # z-1 for 0-based indexing
                    
                    if slices:
                        # Process slices through tiles
                        v = [split_image_into_equal_tiles(slice_data, 2) for slice_data in slices]
                        
                        # Add extra layers if needed (matching original logic)
                        if v:
                            extra_layers = v[-1]
                            if (abs(z - 440)) < 20:
                                for i in range(20 - (abs(z - 440))):
                                    v.append(extra_layers)
                        
                        if v:
                            input_data = np.array(v)
                            
                            # Run through model
                            with torch.no_grad():
                                inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
                                output = model(inp.permute(1, 0, 2, 3, 4).float().to(device))
                                output_value = output.item() * 255
                                output_layer[y, x] = int(np.clip(output_value, 0, 255))
                
                except Exception as e:
                    # Keep original value on error
                    if layer_idx < input_stack.shape[0]:
                        output_layer[y, x] = input_stack[layer_idx, y, x]
                
                pbar.update(1)
    
    return output_layer

def main():
    """Main processing function"""
    
    # Configuration
    input_dir = '../data/dataset_March/NIR_layers_cropped'
    output_dir = 'outputs/clean_stack'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load layers data
    with open('layers_data.txt', "r") as file:
        lines = file.readlines()
    
    print("=== Loading Input Stack ===")
    input_stack = load_input_stack(input_dir)
    
    total_layers = input_stack.shape[0]
    print(f"Processing {total_layers} layers...")
    
    # Initialize output stack
    output_stack = np.zeros_like(input_stack)
    
    # Process each layer
    for layer_idx in range(total_layers):
        try:
            model_axil = int(lines[layer_idx].split()[-1])
            
            output_layer = process_single_layer(layer_idx, input_stack, model_axil)
            
            if output_layer is not None:
                output_stack[layer_idx] = output_layer
                
                # Save individual layer
                layer_image = Image.fromarray(output_layer, mode='L')
                layer_path = os.path.join(output_dir, f"clean_layer_{layer_idx+1:03d}.png")
                layer_image.save(layer_path)
                layer_image.close()
                
                print(f"✓ Completed layer {layer_idx+1}/{total_layers}")
            else:
                print(f"✗ Failed layer {layer_idx+1}")
                # Keep original on failure
                output_stack[layer_idx] = input_stack[layer_idx]
                
        except KeyboardInterrupt:
            print(f"\nStopped at layer {layer_idx+1}")
            break
        except Exception as e:
            print(f"Error processing layer {layer_idx+1}: {e}")
            # Keep original on error
            output_stack[layer_idx] = input_stack[layer_idx]
    
    # Save final stack
    print("=== Saving Results ===")
    
    # Save as numpy array
    stack_path = os.path.join(output_dir, 'clean_stack_440x440x440.npy')
    np.save(stack_path, output_stack)
    print(f"Saved clean stack: {stack_path}")
    
    # Also save as individual PNG files if not already done
    print(f"Individual layers saved in: {output_dir}/")
    
    print("=== Data Generation Complete! ===")
    print(f"Input shape: {input_stack.shape}")
    print(f"Output shape: {output_stack.shape}")

if __name__ == '__main__':
    main()