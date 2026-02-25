import numpy as np
import vtk
from vtk.util import numpy_support
import os
import re
import glob
import cv2
import open3d as o3d
from scipy.interpolate import NearestNDInterpolator

# # Load your point cloud (replace with your file or point cloud data)
# pcd = o3d.io.read_point_cloud(r"/home/haitham/Desktop/25-10-2024/processed_point_cloud_2.0_NIR.ply")

# bbox = pcd.get_axis_aligned_bounding_box()

# points = np.asarray(pcd.points)

# voxel_dimensions = 440

# voxel_indices = points.astype(int)

# # Initialize a voxel grid
# voxel_grid = np.zeros((voxel_dimensions, voxel_dimensions, voxel_dimensions))

# # Fill the voxel grid
# for i, index in enumerate(voxel_indices):
#     if (index[0] < 440 and index[1] < 440 and index [2] < 440) and (index[0] >= 0 and index[1] >= 0 and index [2] >= 0):
#       voxel_grid[tuple(index)[::-1]] = 1

# Load your point cloud (replace with your file or point cloud data)
pcd = o3d.io.read_point_cloud("../../data/dataset_May/RED_colmap_alignment/voxelized.ply")

bbox = pcd.get_axis_aligned_bounding_box()

points = np.asarray(pcd.points) 
colors = np.asarray(pcd.colors) 
# print(points[0])

min_bound = bbox.min_bound
max_bound = bbox.max_bound
normalized_points = (points - min_bound) / (max_bound - min_bound)

voxel_dimensions = 440
scaled_points = normalized_points * (voxel_dimensions - 1)
# Convert to integer voxel coordinates
voxel_indices = scaled_points.astype(int)

# Initialize a voxel grid
voxel_grid_NIR = np.zeros((voxel_dimensions, voxel_dimensions, voxel_dimensions))

print(voxel_indices.shape)
# Fill the voxel grid
t   = 0
for index in voxel_indices:
    voxel_grid_NIR[tuple(index)[::-1]] = 1
    t+=1

voxel_grid_1 = voxel_grid_NIR


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


width, height = 440, 440
num_slices = 1
spacing = .8
img_list = []
img_list2 = []
  
index = 0
for img in sorted(glob.glob('../../data/dataset_May/RED_layers' + '/*.png'),key=numericalSort):
  img_list.append(cv2.imread(img, 0))
  index += 1
  # img_list.append(np.load(img))

combined_image_data = np.concatenate(img_list, axis=0)

# I think similarily you can concatenate multiple data two channel images
####################################
# Normalization for the color bar stretching the color bar to be between 0 - 255
#i = np.where(combined_image_data == combined_image_data.max())
#combined_image_data[i[0][0], i[1][1]] = 255

#i = np.where(combined_image_data == combined_image_data.min())
#combined_image_data[i[0][338], i[1][339]] = 0
####################################

vtk_data = numpy_support.numpy_to_vtk(combined_image_data.reshape(-1, 1), deep=True)
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, height, len(img_list))
vtk_image.SetSpacing(1.0, 1.0, spacing)
# since we are loading the vtk I am not sure if we can change the spacing as you do in paraview but you can set the spacing here instead.
vtk_image.GetPointData().SetScalars(vtk_data)
xx=vtk_image.GetPointData().GetScalars()
vtk_data.SetName('Channels_and_opacity')

# vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  # Assuming 8-bit grayscale images

vtk_data = numpy_support.numpy_to_vtk(voxel_grid_1.reshape(-1, 1), deep=True)
vtk_image.GetPointData().AddArray(vtk_data)
vtk_data.SetName('opacity')




# Write the VTK image data to a .vti file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName('../../data/dataset_May/RED_colmap_alignment/AOS_RED.vti')
writer.SetInputData(vtk_image)
writer.Write()

# Confirmation message
print("Image saved as image.vti")

# reader = vtk.vtkXMLImageDataReader()
# reader.SetFileName(r"/home/haitham/Desktop/25-10-2024/GRE/corrected_GRE.vti")
# reader.Update()

# image_data = reader.GetOutput()
# field_data = image_data.GetFieldData()

# # Extract the transfer function range
# transfer_function_range_array = field_data.GetArray("TransferFunctionRange")
# if transfer_function_range_array:
#     transfer_function_range = [
#         transfer_function_range_array.GetValue(0),
#         transfer_function_range_array.GetValue(1),
#     ]
#     print("Transfer Function Range:", transfer_function_range)
# else:
#     print("Transfer Function Range not found in the file.")