import os
import cv2
import numpy as np
import pycolmap
from .base_module import BaseProcessingModule

class ColmapAlignmentStep(BaseProcessingModule):
    def action(self):
        # Use first channel as reference
        ref_band = f'{self.channel_names[0]}_irradiancee_RGB'
        ref_dir = os.path.join(self.DIR, ref_band, 'cropped')
        
        if not os.path.exists(ref_dir):
            return
            
        ref_images = sorted(os.listdir(ref_dir))
        
        for band in self.channel_names[1:]:
            band_folder = f'{band}_irradiancee_RGB'
            input_dir = os.path.join(self.DIR, band_folder, 'cropped')
            output_dir = os.path.join(self.DIR, band_folder, 'colmap_aligned')
            
            if not os.path.exists(input_dir):
                continue
                
            os.makedirs(output_dir, exist_ok=True)
            
            for img_name in ref_images:
                ref_img_path = os.path.join(ref_dir, img_name)
                img_path = os.path.join(input_dir, img_name)
                
                if not os.path.exists(img_path):
                    continue
                    
                ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Extract features
                sift = cv2.SIFT_create()
                kp1, desc1 = sift.detectAndCompute(ref_img, None)
                kp2, desc2 = sift.detectAndCompute(img, None)
                
                if desc1 is None or desc2 is None:
                    continue
                    
                # Match features
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(desc1, desc2, k=2)
                
                # Filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 10:
                    continue
                    
                # Extract point correspondences
                points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Estimate homography using COLMAP
                ransac_options = pycolmap.RANSACOptions(max_error=4.0)
                result = pycolmap.estimate_homography_matrix(points1, points2, ransac_options)
                
                if result is not None:
                    H = result['H']
                    # Apply homography to align image
                    aligned_img = cv2.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))
                    cv2.imwrite(os.path.join(output_dir, img_name), aligned_img)
