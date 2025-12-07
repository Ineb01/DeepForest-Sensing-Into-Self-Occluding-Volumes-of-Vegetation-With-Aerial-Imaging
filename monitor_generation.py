#!/usr/bin/env python3
"""
Monitor data generation progress
"""
import os
import glob
from PIL import Image
import numpy as np

def check_generation_progress():
    """Check how many layers have been completed"""
    output_dir = 'outputs/clean_stack'
    
    if not os.path.exists(output_dir):
        print("No output directory found")
        return
    
    # Check for completed layers
    pattern = os.path.join(output_dir, 'clean_layer_*.png')
    completed_files = sorted(glob.glob(pattern))
    
    print(f"=== Data Generation Progress ===")
    print(f"Output directory: {output_dir}")
    print(f"Completed layers: {len(completed_files)}/440")
    
    if completed_files:
        # Show first and last completed
        first_file = os.path.basename(completed_files[0])
        last_file = os.path.basename(completed_files[-1])
        
        print(f"First completed: {first_file}")
        print(f"Latest completed: {last_file}")
        
        # Estimate progress
        progress = len(completed_files) / 440 * 100
        print(f"Progress: {progress:.1f}%")
        
        # Check if final stack exists
        stack_file = os.path.join(output_dir, 'clean_stack_440x440x440.npy')
        if os.path.exists(stack_file):
            stack = np.load(stack_file)
            print(f"Final stack available: {stack.shape}")
        
        # Show recent files
        if len(completed_files) > 5:
            print(f"\nRecent layers:")
            for f in completed_files[-5:]:
                print(f"  {os.path.basename(f)}")
    
    else:
        print("No layers completed yet")

def show_sample_layer(layer_num=1):
    """Show information about a specific layer"""
    output_dir = 'outputs/clean_stack'
    layer_file = os.path.join(output_dir, f'clean_layer_{layer_num:03d}.png')
    
    if os.path.exists(layer_file):
        img = Image.open(layer_file)
        img_array = np.array(img)
        
        print(f"\n=== Layer {layer_num} Info ===")
        print(f"File: {layer_file}")
        print(f"Size: {img.size}")
        print(f"Mode: {img.mode}")
        print(f"Array shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        print(f"Value range: {img_array.min()}-{img_array.max()}")
        print(f"Mean value: {img_array.mean():.1f}")
        
        # Show file size
        file_size = os.path.getsize(layer_file)
        print(f"File size: {file_size:,} bytes")
        
        img.close()
    else:
        print(f"Layer {layer_num} not found: {layer_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor data generation progress')
    parser.add_argument('--layer', type=int, help='Show info for specific layer')
    
    args = parser.parse_args()
    
    check_generation_progress()
    
    if args.layer:
        show_sample_layer(args.layer)