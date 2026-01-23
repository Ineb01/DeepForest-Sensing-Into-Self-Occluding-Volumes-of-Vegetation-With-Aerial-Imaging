import cv2
import matplotlib.pyplot as plt
import numpy as np
import os,glob
import sys
from .util.micasense import sequoiautils as msutils
from .util.micasense import metadata as metadata
from .base_module import BaseProcessingModule

class CalibrationStep(BaseProcessingModule):
    def action(self):
        for band in self.channel_names:
            imgs = os.listdir(os.path.join(self.DIR, band))

            for img in imgs:

                imagePath = os.path.join(self.DIR, band)
                imageName = os.path.join(imagePath, img)

                # Read raw image DN values
                # reads 16 / 32 bit tif
                imageRaw=plt.imread(imageName)
                exiftoolPath = None
                if os.name == 'nt':
                    exiftoolPath = 'C:/exiftool/exiftool.exe'
                # get image metadata
                meta = metadata.Metadata(imageName, exiftoolPath=exiftoolPath)
                cameraMake = meta.get_item('EXIF:Make')
                cameraModel = meta.get_item('EXIF:Model')
                firmwareVersion = meta.get_item('EXIF:Software')
                bandName = meta.get_item('XMP:BandName')
                print('{0} {1} firmware version: {2}'.format(cameraMake, 
                                                            cameraModel, 
                                                            firmwareVersion))
                print('Exposure Time: {0} seconds'.format(meta.get_item('EXIF:ExposureTime')))
                print('Imager Gain: {0}'.format(meta.get_item('EXIF:ISO')/100.0))
                print('Size: {0}x{1} pixels'.format(meta.get_item('EXIF:ImageWidth'),meta.get_item('EXIF:ImageHeight')))
                print('Band Name: {0}'.format(bandName))
                print('Center Wavelength: {0} nm'.format(meta.get_item('XMP:CentralWavelength')))
                print('Bandwidth: {0} nm'.format(meta.get_item('XMP:WavelengthFWHM')))
                print('Focal Length: {0}'.format(meta.get_item('EXIF:FocalLength')))

                if bandName != ['Red', 'Green', 'Blue']:
                    SequoiaIrradiance, V = msutils.sequoia_irradiance(meta, imageRaw)

                    # Sunshine sensor Irradiance
                    SunIrradiance = msutils.GetSunIrradiance(meta)
                    print ('Sunshine sensor irradiance: ', SunIrradiance)

                    SequoiaIrradianceCalibrated = SequoiaIrradiance/SunIrradiance

                    if os.path.exists(os.path.join(self.DIR, band+'_irradiancee')):
                        cv2.imwrite(os.path.join(self.DIR, band+'_irradiancee', img), SequoiaIrradianceCalibrated)
                    else:
                        os.mkdir(os.path.join(self.DIR, band+'_irradiancee'))
                        cv2.imwrite(os.path.join(self.DIR, band+'_irradiancee', img), SequoiaIrradianceCalibrated)

                else:
                    if os.path.exists(os.path.join(self.DIR, band+'_irradiancee')):
                        cv2.imwrite(os.path.join(self.DIR, band+'_irradiancee', img), imageRaw)
                    else:
                        os.mkdir(os.path.join(self.DIR, band+'_irradiancee'))
                        cv2.imwrite(os.path.join(self.DIR, band+'_irradiancee', img), imageRaw)