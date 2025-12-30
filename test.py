import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
from slice_contribution import slice_contribution, load_image_stack
import numpy as np
from PIL import Image
import glob
from tools.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

CHECKPOINT_DIR = '../data/checkpoint'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_pixel_value(x, y, model, layer, image_stack):

    if layer <= 420:
        z = layer
        slices = slice_contribution(image_stack, x, y, z, 440, 'slices', 0, False, int(abs(z - 440) // 20))
    else:
        z = layer
        slices = slice_contribution(image_stack, x, y, z, 440, 'slices', 0, False, 1)

    v = [split_image_into_equal_tiles(value, 2) for value in slices]

    extra_layers = v[-1]
    if (abs(z - 440)) < 20 :
        for i in range(20 - (abs(z - 440))):
            v.append(extra_layers)

    input_data = np.array(v)

    with torch.no_grad():
        inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
        test_outputs = model(inp.permute(1, 0, 2, 3, 4).float().to(DEVICE))

    output_value = test_outputs.item() * 255
    return (x, y, output_value) 


def main(layer, model_axil, dataset_dir='', channel=''):
    
    image_dir = os.path.join(dataset_dir, f'{channel}_layers_cropped')
    output_dir = os.path.join(dataset_dir, f'{channel}_layers_cleaned')
    
    os.makedirs(output_dir, exist_ok=True)

    layer = int(layer)
    
    empty_image = Image.new('L', (440, 440), color=(0))
    empty_image = np.zeros_like(empty_image)

    layer_for_model = 160 if layer > 420 else layer

    model_save_path = f'{CHECKPOINT_DIR}/Layer_'+str(layer_for_model)+'.pth'
    
    # Load the model once before the loop
    model = Simple3DCNN(in_channelss=model_axil).to(DEVICE)
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    image_stack = load_image_stack(image_dir, layer_for_model, axil=7)

    width, height = empty_image.shape
    non_zero_pixels = []
    # Collect all pixels to process
    for y in range(height):
        for x in range(width):
            if(x<450 and y<450):  # Limiting to top-left 50x50 for testing
                non_zero_pixels.append(((x, y), 0))

    tqdm.write(f"Processing {len(non_zero_pixels)} pixels...")
    
    for (x,y), _ in tqdm(non_zero_pixels, desc=f"Layer {layer}"):
        x, y, output_value = generate_pixel_value(x, y, model, layer, image_stack)
        empty_image[y, x] = output_value


    # Save the resulting image
    image = Image.fromarray(empty_image, mode='L')
    image.save(f"{output_dir}/Layer_"+str(layer)+".png")
    image.close()


if __name__ == '__main__':
    
    with open('layers_data.txt', "r") as file:
        lines = file.readlines()
        
        dataset_dir = sys.argv[1]
        channel = sys.argv[2]
        layer_start = int(sys.argv[3])
        layer_end = int(sys.argv[4])

        for layer in range(layer_start, layer_end):
            print(int(lines[layer - 1].split()[-1]), layer)
                    
            main(layer, int(lines[layer - 1].split()[-1]), dataset_dir, channel)