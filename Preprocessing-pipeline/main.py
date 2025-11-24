from controllers import MultiSpectralProcessor

from modules import CalibrationStep
from modules import CropBordersStep
from modules import LensCorrectionStep
from modules import ImageAlignmentStep
from modules import SaveGrayscaleStep
from modules import ColmapAlignmentStep

import sys


if __name__ == "__main__":

    # Example image data and reference band
    dataset_path = 'dataset'
    channel_names = ['NIR', 'RED']
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else dataset_path
    print(f"Dataset path: {dataset_path}")
  
    # Create the processing controller with configuration  
    controller = MultiSpectralProcessor(dataset_path, 
                                        channel_names,
                                        #CalibrationStep, 
                                        #SaveGrayscaleStep,
                                        #LensCorrectionStep, 
                                        #ImageAlignmentStep, 
                                        #CropBordersStep,
                                        ColmapAlignmentStep
                                    )

    # Process the images
    final_images = controller.process()
