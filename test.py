import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
from slice_contribution import slice_contribution
import numpy as np
from PIL import Image
import glob
from tools.utils import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor,  as_completed
import pickle
import multiprocessing

def test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer):

    #extra_zeros = np.zeros((2, 2))

    if layer <= 420:
        z = layer
        slices = slice_contribution(x, y, z, 440, 'slices', 0, False, int(abs(z - 440) // 20))
    else:
        z = layer
        slices = slice_contribution(x, y, z, 440, 'slices', 0, False, 1)
    #slices = slice_contribution(x, y, z, 440, 'slices', 0, 5)
    v = [split_image_into_equal_tiles(value, 2) for value in slices]

    extra_layers = v[-1]
    if (abs(z - 440)) < 20 :
        for i in range(20 - (abs(z - 440))):
            v.append(extra_layers)

    input_data = np.array(v)

    with torch.no_grad():
        inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
        test_outputs = model(inp.permute(1, 0, 2, 3, 4).float().to(device))

    output_value = test_outputs.item() * 255
    return (x, y, output_value)  # Return coordinates along with the value

# def process_pixel(pixel_info, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer):
def process_pixel(pixel_info, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer):

    # Load the model inside the process
    model = Simple3DCNN(in_channelss=model_axil)
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    x, y = pixel_info[0]
    return test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer)

def main(layer, model_axil):
    # Use NIR data instead of ground truth for processing
    data_dir = '../data/dataset_March/NIR_layers_cropped'
    layer = int(layer)
    
    # Get NIR layer image for processing
    layer_files = sorted(glob.glob(data_dir + '/*.png'), key=numericalSort)
    if layer - 1 < len(layer_files):
        layer_img = Image.open(layer_files[layer - 1]).convert('L')
        print(f"Processing layer: {layer_files[layer - 1]}")
    else:
        print(f"Layer {layer} not found in data directory")
        return
    
    # Create output matrix based on image dimensions
    width, height = layer_img.size
    random_matrix = np.zeros((height, width))

    layer = 160 if layer > 420 else layer

    model_save_path = f'../data/checkpoint/Layer_'+str(layer)+'.pth'
    
    device = torch.device('cpu')  # Use CPU on Mac since Conv3D not supported on MPS
    all_pixels = []

    # Process all pixels (no ground truth comparison needed)
    for y in range(height):
        for x in range(width):
            all_pixels.append(((x, y), 255))  # Use dummy value since we're not comparing

    # Parallel processing of all pixels using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_pixel, i, model_save_path, layer_img, random_matrix, device, model_axil, layer) for i in all_pixels]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            try:
                x, y, output_value = future.result()
                random_matrix[y, x] = output_value
            except Exception as exc:
                print(f'Error occurred: {exc}')

    # Save the resulting image
    image = Image.fromarray(random_matrix.astype(np.uint8), mode='L')
    os.makedirs("outputs", exist_ok=True)
    image.save(f"outputs/Layer_{layer}_processed.png")
    image.close()

def load_image_stack(directory, image_type, layer_number, axil = 1):
    # Pattern to match the desired image files
    pattern = os.path.join(directory, '*.png')
    imgs = []
    # Sorted list of matching files
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    file_list = file_list[layer_number - 1::axil]

    for filename in file_list:
        # Load the image and convert to grayscale ('L')
        img = Image.open(filename).convert('L')
        image_array = np.array(img)
        imgs.append(image_array)

    return np.array(imgs)


if __name__ == '__main__':
    with open('layers_data.txt', "r") as file:
        lines = file.readlines()

    image_stack_dir = '../data/dataset_March/NIR_layers_cropped'
    device = torch.device('cpu')  # Use CPU on Mac since Conv3D not supported on MPS

    for layer in range(300, 301):
        print(int(lines[layer - 1].split()[-1]), layer)
                
        main(layer, int(lines[layer - 1].split()[-1]))
