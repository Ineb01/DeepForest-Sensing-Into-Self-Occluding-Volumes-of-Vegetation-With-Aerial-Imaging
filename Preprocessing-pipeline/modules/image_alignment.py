import os 
import cv2 
import numpy as np 
from .base_module import BaseProcessingModule

class ImageAlignmentStep(BaseProcessingModule):
    def action(self):
        # Implement code to run 'align_images.ipynb' notebook to align the images.
        # This would align all bands with a selected reference band.
        DIR = self.DIR
        ref_imgs = os.listdir(os.path.join(DIR, f'{self.channel_names[0]}_irradiancee_RGB', 'undistord_solving'))
        bands = [f'{band}_irradiancee_RGB' for band in self.channel_names[1:]]

        for img in ref_imgs:
            # Open the image files. 
            img2 = cv2.imread(os.path.join(DIR, f'{self.channel_names[0]}_irradiancee_RGB', 'undistord_solving', img), -1) #  Reference image
            for band in bands:        
                # Replace band name in filename
                target_img = img.replace(self.channel_names[0], band.replace('_irradiancee_RGB', ''))
                    
                img1 = cv2.imread(os.path.join(DIR, band, 'undistord_solving', target_img), -1) # Image to be aligned. 
                
                # Initiate SIFT detector
                sift_detector = cv2.SIFT_create()
                # Find the keypoints and descriptors with SIFT
                kp1, des1 = sift_detector.detectAndCompute(img1, None)
                kp2, des2 = sift_detector.detectAndCompute(img2, None)

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                # Filter out poor matches
                good_matches = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good_matches.append(m)

                matches = good_matches
                        
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt

                # Find homography
                H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

                # Warp image 1 to align with image 2
                img1Reg = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                
                # Create align directory if it doesn't exist
                align_dir = os.path.join(DIR, band, 'align')
                os.makedirs(align_dir, exist_ok=True)
                
                name = target_img.split('.')[0][:-3]
                cv2.imwrite(os.path.join(align_dir, name+band.replace('_irradiancee_RGB', '')+'.jpg'), img1Reg)
