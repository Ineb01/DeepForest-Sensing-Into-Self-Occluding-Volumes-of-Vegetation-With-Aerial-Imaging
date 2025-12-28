from PIL import Image
import os
from .base_module import BaseProcessingModule

class RotateStep(BaseProcessingModule):
    def action(self):
        bands = [f'{band}_irradiancee_RGB' for band in self.channel_names]
        for band in bands:
            img_dir = os.path.join(self.DIR, band, 'cropped')
            
            if not os.path.exists(img_dir):
                print(f"Directory not found, skipping: {img_dir}")
                continue
                
            imgs = os.listdir(img_dir)
            imgs.sort()
            print(f"Rotating images in {img_dir}")
            for img_name in imgs:
                img_path = os.path.join(img_dir, img_name)
                try:
                    with Image.open(img_path) as img:
                        rotated_img = img.rotate(180)
                        rotated_img.save(img_path)
                except Exception as e:
                    print(f"Could not process {img_path}: {e}")
