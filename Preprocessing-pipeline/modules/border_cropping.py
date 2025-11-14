from PIL import Image
import os
from PIL import ImageFile

class CropBordersStep:
    def __init__(self, output_directory, channel_names):
        self.DIR = output_directory
        self.channel_names = channel_names


    def crop_borders(self):
        # Implement code to crop the black borders from aligned images.
        # You can use OpenCV or similar libraries to perform cropping.
        bands = [f'{band}_irradiancee_RGB' for band in self.channel_names]
        for band in bands:
            if band == f'{self.channel_names[1]}_irradiancee_RGB':
                imgs = os.listdir(os.path.join(self.DIR, band, 'undistord_solving'))
            else:
                imgs = os.listdir(os.path.join(self.DIR, band, 'align'))
                
            imgs.sort()
            for img in imgs:
                if band == f'{self.channel_names[1]}_irradiancee_RGB':
                    rgb_img2 = Image.open(os.path.join(self.DIR, band, 'undistord_solving', img))
                else:
                    rgb_img2 = Image.open(os.path.join(self.DIR, band, 'align', img))


                cropped_rgb_image2 = rgb_img2.crop((50, 50, rgb_img2.width - 50, rgb_img2.height - 50))
                
                if os.path.exists(os.path.join(self.DIR, band, 'cropped')):
                    print("Folder exists")
                else:
                    os.mkdir(os.path.join(self.DIR, band, 'cropped'))
                    
                cropped_rgb_image2.save(os.path.join(self.DIR, band, 'cropped', img))
