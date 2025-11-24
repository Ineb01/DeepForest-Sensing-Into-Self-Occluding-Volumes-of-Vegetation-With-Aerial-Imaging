import os
import cv2
import numpy as np
import pycolmap
from .base_module import BaseProcessingModule

class ColmapAlignmentStep(BaseProcessingModule):
    def action(self):
        # Implement COLMAP-based alignment for multispectral images
        for band in self.channel_names[1:]:
            band_folder = f'{band}_irradiancee_RGB'
            input_dir = os.path.join(self.DIR, band_folder, 'cropped')
            output_dir = os.path.join(self.DIR, f'{band}_colmap_alignment')
            database_path = os.path.join(output_dir, f'{band}_colmap.db')
            
            if not os.path.exists(input_dir):
                print(f"Input directory {input_dir} does not exist, skipping band {band}")
                continue
            
            #if os.path.exists(database_path):
            #    os.unlink(database_path)
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"Processing band {band}, saving aligned images to {output_dir}")
            
            pycolmap.extract_features(
                database_path,
                input_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model='SIMPLE_PINHOLE',
            )
            
            pycolmap.match_exhaustive(
                database_path,
                matching_options={'max_num_matches': 65536,'num_threads': -1},
                pairing_options={'block_size': 27},
                # use_gpu=True currently only works with python module compiled from source with cuda
            )
           
            pycolmap.incremental_mapping(
                database_path,
                image_path = input_dir,
                output_path = os.path.join(output_dir, "bin_output")
            )
            
            # import from output_path
            reconstructions = pycolmap.Reconstruction(path=os.path.join(output_dir, "bin_output/0"))
            
            os.makedirs(os.path.join(output_dir, "output"), exist_ok=True)
            reconstructions.write_text(os.path.join(output_dir, "output"))
            
            #if os.path.exists(database_path):
            #    os.unlink(database_path)
            
            
