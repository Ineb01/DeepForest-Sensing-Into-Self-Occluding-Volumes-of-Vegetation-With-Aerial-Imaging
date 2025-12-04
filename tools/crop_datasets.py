from PIL import Image
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset folders to process
datasets = ['dataset_March', 'dataset_May', 'dataset_October']
base_path = '/mnt/c/Users/benja/Desktop/git/university/master-thesis/practical/data'

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    
    # Process NIR_layers
    nir_input_path = os.path.join(dataset_path, 'NIR_layers')
    nir_output_path = os.path.join(dataset_path, 'NIR_layers_cropped')
    
    if os.path.exists(nir_input_path):
        os.makedirs(nir_output_path, exist_ok=True)
        print(f"Processing {dataset}/NIR_layers...")
        
        for img_file in os.listdir(nir_input_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(nir_input_path, img_file)
                img = Image.open(img_path)
                
                # Crop the image (remove 260 pixels from each side)
                cropped_img = img.crop((260, 260, img.width - 260, img.height - 260))
                
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
        
        for img_file in os.listdir(red_input_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(red_input_path, img_file)
                img = Image.open(img_path)
                
                # Crop the image (remove 260 pixels from each side)
                cropped_img = img.crop((260, 260, img.width - 260, img.height - 260))
                
                # Save cropped image
                output_path = os.path.join(red_output_path, img_file)
                cropped_img.save(output_path)
        
        print(f"Completed {dataset}/RED_layers - {len(os.listdir(red_output_path))} images cropped")

print("All datasets processed successfully!")
