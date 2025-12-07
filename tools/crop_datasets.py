from PIL import Image
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset folders to process
datasets = ['dataset_March', 'dataset_May', 'dataset_October']
base_path = '../../data'  # Mac path (from tools/ directory)

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    print(f"Checking dataset: {dataset_path}")
    
    # Process NIR_layers
    nir_input_path = os.path.join(dataset_path, 'NIR_layers')
    nir_output_path = os.path.join(dataset_path, 'NIR_layers_cropped')
    
    print(f"  NIR input path: {nir_input_path} (exists: {os.path.exists(nir_input_path)})")
    
    if os.path.exists(nir_input_path):
        os.makedirs(nir_output_path, exist_ok=True)
        print(f"Processing {dataset}/NIR_layers...")
        
        files = [f for f in os.listdir(nir_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(files)} image files")
        
        for i, img_file in enumerate(files):
                img_path = os.path.join(nir_input_path, img_file)
                img = Image.open(img_path)
                
                # Step 1: Downscale to 512x512
                downscaled_img = img.resize((512, 512), Image.LANCZOS)
                
                # Step 2: Crop center 440x440 from the 512x512 image
                center_x, center_y = 256, 256  # Center of 512x512
                crop_size = 220  # Half of 440
                left = center_x - crop_size
                top = center_y - crop_size
                right = center_x + crop_size
                bottom = center_y + crop_size
                cropped_img = downscaled_img.crop((left, top, right, bottom))
                
                # Debug: Show sizes for first few images
                if i < 3:
                    print(f"    {img_file}: {img.width}x{img.height} -> 512x512 -> {cropped_img.width}x{cropped_img.height}")
                
                # Save cropped image
                output_path = os.path.join(nir_output_path, img_file)
                cropped_img.save(output_path)
        
        print(f"Completed {dataset}/NIR_layers - {len(os.listdir(nir_output_path))} images cropped")
    
    # Process RED_layers
    red_input_path = os.path.join(dataset_path, 'RED_layers')
    red_output_path = os.path.join(dataset_path, 'RED_layers_cropped')
    
    if os.path.exists(red_input_path):
        os.makedirs(red_output_path, exist_ok=True)
        print(f"Processing {dataset}/RED_layers...")
        
        files = [f for f in os.listdir(red_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(files)} image files")
        
        for i, img_file in enumerate(files):
                img_path = os.path.join(red_input_path, img_file)
                img = Image.open(img_path)
                
                # Step 1: Downscale to 512x512
                downscaled_img = img.resize((512, 512), Image.LANCZOS)
                
                # Step 2: Crop center 440x440 from the 512x512 image
                center_x, center_y = 256, 256  # Center of 512x512
                crop_size = 220  # Half of 440
                left = center_x - crop_size
                top = center_y - crop_size
                right = center_x + crop_size
                bottom = center_y + crop_size
                cropped_img = downscaled_img.crop((left, top, right, bottom))
                
                # Debug: Show sizes for first few images
                if i < 3:
                    print(f"    {img_file}: {img.width}x{img.height} -> 512x512 -> {cropped_img.width}x{cropped_img.height}")
                
                # Save cropped image
                output_path = os.path.join(red_output_path, img_file)
                cropped_img.save(output_path)
        
        print(f"Completed {dataset}/RED_layers - {len(os.listdir(red_output_path))} images cropped")

print("All datasets processed successfully!")
