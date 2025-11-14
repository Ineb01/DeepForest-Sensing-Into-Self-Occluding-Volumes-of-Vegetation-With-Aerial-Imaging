import os
import sys
# print(os.getcwd())  # Print the current working directory
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))



class MultiSpectralProcessor:
    def __init__(self, dataset_path, channel_names, Calibration, ImageSaver, LensCorrection, ImageAlignment, BorderCropping):
        
     
        self.dataset_path = dataset_path
        self.channel_names = channel_names
        # Calibration takes the dataset path
        self.calibration_step = Calibration(dataset_path, channel_names)
        
        self.save_grayscale_step = ImageSaver(dataset_path, channel_names)
        
        self.lens_correction_step = LensCorrection(dataset_path, channel_names)
        self.alignment_step = ImageAlignment(dataset_path, channel_names)
        self.crop_step = BorderCropping(dataset_path, channel_names)

    def process(self):
        # Step 1: Perform Calibration
        self.calibration_step.perform_calibration()

        # # Step 2: Save outputs in grayscale
        self.save_grayscale_step.save_as_grayscale()

        # Step 3: Correct Lens Distortion
        self.lens_correction_step.correct_lens_distortion()

        # Step 4: Align the Images
        self.alignment_step.align_images()

        # Step 5: Crop Black Borders
        cropped_images = self.crop_step.crop_borders()

        # return cropped_images
