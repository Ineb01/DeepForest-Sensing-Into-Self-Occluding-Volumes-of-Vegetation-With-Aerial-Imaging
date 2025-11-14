
import numpy as np
import math
import os
import cv2 as cv2
import glob
import piexif

class LensCorrectionStep:
    """Class for image undistortion using OpenCV's `UndistortRectifyMap`

    :param new_size: new image resolution of the undistored images, defaults to (512,512)
    :type new_size: tuple, optional

    :param f_factor: focal length factor, defaults to .79
    :type f_factor: float, optional

    :param camera_type: camera name defining the calibrated camera parameters (distorion coefficients and camera matrix),
        defaults to 'framegrabber_HDMI_fixed_tangential_k3'
    :type camera_type: str, optional
    """
    mapx = None
    mapy = None
    _f_factor = None

    # constructor
    def __init__(self, output_directory, channel_names, new_size = (1280,960), f_factor = .912, camera_type = "seqouia_parrot_GREEN"):
        """Constructor method
        """
        self.DIR = output_directory
        self.channel_names = channel_names
        
        self._f_factor = f_factor
        #// focal length as parameter:
        #//     f-factor of 4 -> 126.8698976458440 degrees
        #//     f-factor of 2 -> 90.0 degrees
        #//     f-factor of 1 -> 53.1301 degrees
        #//     f-factor of .95 -> 50.815436217896945 degrees (0.886896672839476 rad)
        #//     f-factor of .79 -> 43.10803984095769 degrees
        #
        # f-factor to degrees equation: 2*math.degrees( math.atan2( f_factor / 2.0, 1.0 ) )


        px = new_size[0]/2.0
        py = new_size[1]/2.0
        fx = new_size[0]/f_factor
        fy = new_size[1]/f_factor

        new_K = np.array( [
                [fx, 0, px],
                [0, fy, py],
                [0, 0, 1.0]
            ] )

        R = np.array( [
                [1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]
            ] )
        print(camera_type)
        if camera_type == "framegrabber_analog": # full optimization with 5 distortion coefficients (including tangential distortions ...)
            distCoeff = np.array([
                    -3.0157272169483829e-01, 1.6762278444125270e-01,
                    -5.9877119825392754e-03, 2.5697912533955846e-04,
                    3.4034370939765371e-02 
                ])
            cameraMatrix = np.array( [
                    [ 4.0654113564528342e+02, 0., 2.3405342412922565e+02,],
                    [ 0.,    3.5876940096730635e+02, 1.3891173431827505e+02,],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "framegrabber_analog_fixed_tangential": ## only 3 distortion parameters
            distCoeff = np.array([ -3.0931730982886513e-01, 1.7651094292715205e-01, 0., 0.,
                    1.3892581368875190e-02 ])
            cameraMatrix = np.array( [
                    [ 4.0827067154798226e+02, 0., 2.3158065054354975e+02],
                    [ 0.,    3.5823788014881160e+02, 1.4076564463687103e+02,],
                    [ 0., 0., 1.0 ]
                ])
            
        elif camera_type == "framegrabber_analog_fixed_tangential_k3": # only 2 distortion parameters
            distCoeff = np.array([ -3.1038440439237142e-01, 1.8380464680292599e-01, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 4.0828175759937125e+02, 0., 2.3153629709067536e+02],
                    [ 0.,    3.5824063978980092e+02, 1.4075493169694383e+02],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "seqouia_parrot_NIR": # only 2 distortion parameters
            distCoeff = np.array([ -0.361103326348155, 0.160484696910165, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1.091625508496761e+03, 0., 5.342123310421389e+02],
                    [ 0.,    1.088751761560764e+03, 4.164781958014993e+02],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "seqouia_parrot_GREEN": # only 2 distortion parameters
            distCoeff = np.array([ -0.359475743425442, 0.160049484238977, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1092.36795000265, 0., 527.283929159181],
                    [ 0.,    1088.52031774928, 420.938564073592],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "seqouia_parrot_RED": # only 2 distortion parameters
            distCoeff = np.array([ -0.357683388824061, 0.155131772456130, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1089.82581780467, 0., 521.133368614276],
                    [ 0.,    1087.10104693055, 414.922411908088],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "seqouia_parrot_REG": # only 2 distortion parameters
            distCoeff = np.array([ -0.359051752113787, 0.1572316893989485, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1088.08225743108, 0., 527.106393009359],
                    [ 0.,    1085.27208023758, 428.940989850558],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "seqouia_parrot_RGB": # only 2 distortion parameters
            distCoeff = np.array([ 0.193975872489218, -0.390825317340173, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 3690.17462547856, 0., 2307.59008737449],
                    [ 0.,    3673.32056050602, 1697.59465638679],
                    [ 0.,           0.,             1.0 ]
                ])
        # use below, for HDMI Framegrabber, per default use .79 as f-factor
        elif camera_type == "framegrabber_HDMI_fixed_tangential_k3": # only 2 distortion parameters
            distCoeff = np.array([ -0.2536, 0.0649, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 417.8933, 0., 344.4168],
                    [ 0.,   526.2962, 206.0617],
                    [ 0., 0., 1.0 ]
                ])
        # use below, for DSC-SPYCameras
        elif camera_type == "DSC_Camera_G01": # only 2 distortion parameters
            distCoeff = np.array([ -0.0201, 0.1030, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1542.4, 0., 937.5931],
                    [ 0.,   1545.2, 510.3446],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_G02": # only 2 distortion parameters
            distCoeff = np.array([ 0.0043, 0.7874, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1309.5, 0., 664.7523],
                    [ 0.,   1545.2, 510.3446],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_G03": # only 2 distortion parameters
            distCoeff = np.array([ 0.0835, 0.4993, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1351.3, 0., 702.4611],
                    [ 0.,   1352.8, 321.5063],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_G04": # only 2 distortion parameters
            distCoeff = np.array([ 0.0827, 1.3024, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1368.4, 0., 671.0829],
                    [ 0.,   1369.1, 323.0925],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_G05": # only 2 distortion parameters
            distCoeff = np.array([ 0.0152, 0.9354, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1290.5, 0., 655.4665],
                    [ 0.,   1292.1, 408.2471],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_B01": # only 2 distortion parameters
            distCoeff = np.array([ 0.0257, 0.7085, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1325.9, 0., 681.7295],
                    [ 0.,   1323.5, 362.3177],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_B02": # only 2 distortion parameters
            distCoeff = np.array([ -0.0209, 0.4680, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1289.7, 0., 665.7060],
                    [ 0.,   1287.7, 366.5962],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_B03": # only 2 distortion parameters
            distCoeff = np.array([ -0.0072, 0.4956, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1307.7, 0., 663.2968],
                    [ 0.,   1304.2, 348.0411],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_B04": # only 2 distortion parameters
            distCoeff = np.array([ -0.0072, 1.6172, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1349.8, 0., 683.0524],
                    [ 0.,  1348.1, 400.5760],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_B05": # only 2 distortion parameters
            distCoeff = np.array([ -0.3057, 0.8105, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 1364.7, 0., 597.9454],
                    [ 0.,   1353.6, 186.7618],
                    [ 0., 0., 1.0 ]
                ])
        elif camera_type == "DSC_Camera_Alls": # only 2 distortion parameters
            distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            #distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 957.9752, 0., 634.8713],                                         #old calibration data
                   [ 0.,   970.8227, 351.3978],
                   [ 0., 0., 1.0 ]
               ])
            
        elif camera_type == "DSC_Camera_Alltesting": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 951.462, 0, 639.55],                                         #testing
                   [ 0., 964.579, 362.738],
                   [ 0., 0., 1.0 ]
               ])    
        elif camera_type == "DSC_Camera_All3line": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [                                                   #threelinescan
                   [ 957.9752, 0., 650.8713],
                   [ 0.,   970.8227, 351.3978],
                   [ 0., 0., 1.0 ]
               ])    
        elif camera_type == "DSC_Camera_All2line": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])   #1280x720            #04.05.2022  2line scan
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 957.9752, 0., 628],
                    [ 0.,   970.8227, 360],
                    [ 0., 0., 1.0 ]
                ])  
            
        elif camera_type == "DSC_Camera_All2linecalibavg": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])   #1280x720            #04.05.2022  2line scan calib avg
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                    [ 951.462, 0.408143002, 639.355],
                    [ 0.,   964.579, 362.738],
                    [ 0., 0., 1.0 ]
                ])     
        elif camera_type == "DSC_Camera_Allset1": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 977.2438, 0., 640.3841],                                #set1 calibration (0.0094,0.0174) # drone1
                   [ 0.,   989.3205, 363.4386],
                   [ 0., 0., 1.0 ]
               ]) 
            
        elif camera_type == "DSC_Camera_Allset2": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 970.4858, 0., 642.2818],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,   982.2169, 361.4631],
                   [ 0., 0., 1.0 ]
               ])     
            
        elif camera_type == "DSC_Camera_calib2": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ -0.0127111799353962, 0.0383346583582007, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 951.681982816154, 0., 641.446542935422],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,  963.782699882553, 355.283755447403],
                   [ 0., 0., 1.0 ]
               ])  
            
        elif camera_type == "DSC_Camera_calib4": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 954.388219, 0.494662697330557, 634.8743],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,   969.32507, 375.51492],
                   [ 0., 0., 1.0 ]
               ])     
            
        elif camera_type == "DSC_Camera_calib1": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 951.704348, 0.323186492949581, 639.964489],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,   964.4006156, 356.442842],
                   [ 0., 0., 1.0 ]
               ]) 
            
        elif camera_type == "DSC_Camera_calib3": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 954.240359, 0.842323368938120, 643.605057],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,   966.542323, 356.069199],
                   [ 0., 0., 1.0 ]
               ])  
            
        elif camera_type == "DSC_Camera_calib5": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.0, 0.0, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 945.295409148025, -0.255749227371709, 636.885874812461],                                #set2 calibration (-0.0635,0.4989)
                   [ 0.,   958.844191849244, 370.378556232893],
                   [ 0., 0., 1.0 ]
               ])   
            
        elif camera_type == "MAVIC_drone1_stereo": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.207875797879643, -0.512734263215905, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 1418.71331034629, 0, 963.793992167989],                             
                   [ 0.,   1413.47965430266, 519.698431089368],
                   [ 0., 0., 1.0 ]
               ]) 
            
        elif camera_type == "MAVIC_drone2_stereo": # only 2 distortion parameters
            #distCoeff = np.array([ -0.0418, 0.2965, 0., 0., 0.  ])
            distCoeff = np.array([ 0.277946590805523, -0.784404423078403, 0., 0., 0.  ])
            cameraMatrix = np.array( [
                   [ 1423.45305560630, 0, 947.296042408568],                                
                   [ 0.,   1415.85735755769, 520.066384715638],
                   [ 0., 0., 1.0 ]
               ]) 
                        
        else: # FLIR TIFF images (~640x512) use .95 as f-factor
            print( 'Undistort: USING default parameters!')
            distCoeff = np.array([ -2.8637715386958607e-001, 2.0357125936656664e-001,
                    1.5036407221462624e-003, 8.5758458509730892e-004, -2.9228054407644311e-001 
                ])
            cameraMatrix = np.array( [
                    [ 5.3363720699684154e+002, 0., 3.1659760411813903e+002],
                    [ 0.,    5.3438292680299116e+002, 2.5963215645943137e+002],
                    [ 0., 0., 1.0 ]
                ])
        

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeff,
            R=R,
            newCameraMatrix=new_K,
            size=new_size,
            m1type=cv2.CV_32FC1)

    def undistort(self, img ):
        """undistorts an image with OpenCV's builtin functions and returns it

        :param img: image with lens distortions
        :type img: opencv image as numpy array

        :return: an undistorted image without any lens distoritions and cropped to the specified field of view
        :rtype: opencv image as numpy array

        """
         # in C++ -> cv::remap(distorted, undistorted, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);   
        img_rect = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR | cv2.BORDER_CONSTANT)

        return img_rect

    def getFocalLengthFactor(self):
        """returns the focal-length factor

        :return: focal length factor
        :rtype: float
        """
        return self._f_factor

    def getFieldOfViewInDegrees(self):
        """returns the field-of-view angle degrees

        :return: angle in degrees
        :rtype: float
        """
        return 2*math.degrees( math.atan2( self.getFocalLengthFactor() / 2.0, 1.0 ) )

    def correct_lens_distortion(self):
        # Implement code to run the 'Undistort_RGB.py' script programmatically on the grayscale images.
        # This could involve using subprocess to execute the script or using OpenCV methods directly in Python.

        # img_size = (4608, 3456)
        img_size = (1280,960)
        # img_size = (1280,960)
        # dsc_camera_type = ['seqouia_parrot_REG', 'seqouia_parrot_RED', 'seqouia_parrot_NIR', 'seqouia_parrot_GREEN'] 
        # image_folder = ['REG_irradiancee_RGB', 'RED_irradiancee_RGB', 'NIR_irradiancee_RGB', 'GRE_irradiancee_RGB']

        dsc_camera_type = [f'seqouia_parrot_{band}' for band in self.channel_names] 
        image_folder = [f'{band}_irradiancee_RGB' for band in self.channel_names]
        # convertion between degree and radian
        # 0,855211 -- 49
        # 0.872665 -- 50
        for i, camera_type in enumerate(dsc_camera_type):
            ud = LensCorrectionStep(self.DIR, new_size=img_size,f_factor = 0.855211,camera_type=camera_type)

            images_path = os.path.join(self.DIR, image_folder[i], 'images') 
            
            if os.path.exists(os.path.join(self.DIR, image_folder[i], 'undistord_solving')):
                print("Folder exists")
            else:
                os.mkdir(os.path.join(self.DIR, image_folder[i], 'undistord_solving'))
 
            images_save_path = os.path.join(self.DIR, image_folder[i], 'undistord_solving') 
            rectify_path = os.path.join(images_save_path)
            if rectify_path and not os.path.isdir(rectify_path):
                os.mkdir( rectify_path )
            for filename in glob.glob(os.path.join(images_path,'*.JPG')): #assuming
                png_filename = os.path.splitext(os.path.basename(filename))[0] + '.JPG'
                img = cv2.imread(os.path.join(filename) )
                exif_dict = piexif.load(os.path.join(filename) )
                
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # make sure the image is grayscale
                und = ud.undistort(img)
                cv2.imwrite(os.path.join( rectify_path, os.path.basename(png_filename) ), und)
                # Write the metadata
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, os.path.join( rectify_path, os.path.basename(png_filename) ))