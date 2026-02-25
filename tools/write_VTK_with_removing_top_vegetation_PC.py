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

# # Load your point cloud (replace with your file or point cloud data)
# pcd = o3d.io.read_point_cloud(r"/home/haitham/Desktop/25-10-2024/processed_point_cloud_2.0_NIR.ply")

# bbox = pcd.get_axis_aligned_bounding_box()

# points = np.asarray(pcd.points) 
# colors = np.asarray(pcd.colors) 
# # print(points[0])

# min_bound = bbox.min_bound
# max_bound = bbox.max_bound
# normalized_points = (points - min_bound) / (max_bound - min_bound)

# voxel_dimensions = 440
# scaled_points = normalized_points * (voxel_dimensions - 1)
# # Convert to integer voxel coordinates
# voxel_indices = scaled_points.astype(int)

# # Initialize a voxel grid
# voxel_grid_NIR = np.zeros((voxel_dimensions, voxel_dimensions, voxel_dimensions))

# print(voxel_indices.shape)
# # Fill the voxel grid
# t   = 0
# for index in voxel_indices:
#     voxel_grid_NIR[tuple(index)] = np.mean(colors[t])
#     t+=1

# voxel_grid_1 = voxel_grid_NIR

#####################################################################################################
# Load your point cloud (replace with your file or point cloud data)
pcd1 = o3d.io.read_point_cloud(r"../../data/dataset_October/RED_colmap_alignment/voxelized.ply")

bbox1 = pcd1.get_axis_aligned_bounding_box()

points1 = np.asarray(pcd1.points) 
colors1 = np.asarray(pcd1.colors) 

min_bound1 = bbox1.min_bound
max_bound1 = bbox1.max_bound
normalized_points1 = (points1 - min_bound1) / (max_bound1 - min_bound1)

voxel_dimensions1 = 440
scaled_points1 = normalized_points1 * (voxel_dimensions1 - 1)
# Convert to integer voxel coordinates
voxel_indices1 = scaled_points1.astype(int)

# Initialize a voxel grid
voxel_grid_RED = np.zeros((voxel_dimensions1, voxel_dimensions1, voxel_dimensions1))

# Fill the voxel grid
t   = 0

for index in voxel_indices1:
    voxel_grid_RED[tuple(index)] = np.mean(colors1[t])
    t+=1
    # print(voxel_grid_RED[tuple(index)[::-1]], tuple(index)[::-1])


# voxel_grid_2 = voxel_grid_RED

#########################################################################################

# mergedPC = np.zeros((voxel_dimensions, voxel_dimensions, voxel_dimensions))
# x=0
# for i in range(440):
#     for j in range(440):
#         for k in range(440):
#             if voxel_grid_1[i,j,k] or voxel_grid_2[i,j,k]:
#             # if voxel_grid_1[i,j,k]:
#                 x+=1
#                 # print(voxel_grid_1[i,j,k] , voxel_grid_2[i,j,k])
#                 ii = (i,j,k)
#                 # mergedPC[tuple(ii)[::-1]] = 1
#                 # mergedPC[tuple(ii)[::-1]] = 1
#                 mergedPC[tuple(ii)] = 1

# Use only RED point cloud
mergedPC = voxel_grid_RED
depth_map = np.zeros((440, 440))
depth_map2 = np.ones((440, 440)) * -1
depth_3d = np.zeros((440, 440, 440))

for i in range(440):
    for j in range(440):
        for k in range(440):
            if mergedPC[i,j,k]:
                ii = (i,j)
                depth_map[tuple(ii)[::-1]] = k+1
                depth_map2[tuple(ii)[::-1]] = k+1
                continue

# Step 1: Find the indices of valid (non-NaN) and missing (NaN) values
x, y = np.indices(depth_map.shape)
valid_points = np.column_stack((x[depth_map != 0], y[depth_map != 0]))  # Indices of non-NaN values
missing_points = np.column_stack((x[depth_map == 0], y[depth_map == 0]))  # Indices of NaN values
valid_values = depth_map[depth_map != 0]  # Non-NaN values

# Step 2: Use Nearest Neighbor Interpolation
interpolator = NearestNDInterpolator(valid_points, valid_values)
filled_values = interpolator(missing_points)

# Step 3: Fill the 
filled_matrix = depth_map.copy()
filled_matrix[depth_map == 0] = filled_values

depth_map = filled_matrix

########################################################################################################

max_depth_map = np.max(depth_map)
depth_3d = np.zeros((440, 440, 440))

for z in range(1, int(max_depth_map)+1):
    depth_3d[z-1] = (depth_map >= z).astype(int)

########################################################################################################

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
depth_map3 = np.ones((440, 440)) * -1.1
index = 0

for img in sorted(glob.glob('../../data/dataset_October/NDVI_layers' + '/*.npy'),key=numericalSort):
  ll_list = []
  indices = np.where(depth_3d[index] == 0)
  ndvi_layer = np.load(img)  # 2D array (440x440)
  index_0 = ndvi_layer * depth_3d[index]
  index_0[indices] = np.nan
  index_1 = index_0
  index_2 = np.zeros_like(index_0)  # Third channel as zeros

  ll_list = np.stack((index_0, index_1, index_2), axis=-1)
  img_list.append(ll_list)
  img_list2.append(index_0)
  index += 1
  # img_list.append(np.load(img))

for i in range(440):
    for j in range(440):
        # for k in range(440):
        if int(depth_map2[i,j]) > 0 and int(depth_map2[i,j]) < 440:
          ii = (i,j)
          depth_map3[tuple(ii)] = img_list2[int(depth_map2[i,j])-1][i,j]
          # continue
          
from PIL import Image

#Image.fromarray(index_1.astype(np.float32)).save("/home/haitham/Desktop/25-10-2024/NDVI-below-canopz/"+str(index)+".tiff")

# Image.fromarray(depth_map3.astype(np.float32)).save("/home/haitham/Desktop/25-10-2024/nnddvvii.tiff")


combined_image_data = np.concatenate(img_list, axis=0)


# I think similarily you can concatenate multiple data two channel images

vtk_data = numpy_support.numpy_to_vtk(combined_image_data.reshape(-1, 3), deep=True)

vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, height, len(img_list))
vtk_image.SetSpacing(1.0, 1.0, spacing)
# since we are loading the vtk I am not sure if we can change the spacing as you do in paraview but you can set the spacing here instead.
vtk_image.GetPointData().SetScalars(vtk_data)
vtk_data.SetName('Channels_and_opacity')

# vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  # Assuming 8-bit grayscale images

# vtk_data = numpy_support.numpy_to_vtk(voxel_grid.reshape(-1, 1), deep=True)
vtk_data = numpy_support.numpy_to_vtk(depth_3d.reshape(-1, 1), deep=True)
vtk_image.GetPointData().AddArray(vtk_data)
vtk_data.SetName('opacity')


# Write the VTK image data to a .vti file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(r'../../data/dataset_October/corrected_NDVI_new.vti')
writer.SetInputData(vtk_image)
writer.Write()

# Confirmation message
print("Image saved as image.vti")