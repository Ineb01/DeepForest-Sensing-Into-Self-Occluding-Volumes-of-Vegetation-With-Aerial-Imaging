import open3d as o3d
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw


# Load your point cloud (replace with your file or point cloud data)
pcd = o3d.io.read_point_cloud(r"../../data/dataset_May/RED_colmap_alignment/dense/fused_cut.ply")

# Voxel size
# voxel_size = 0.01  # Change this value as needed

bbox = pcd.get_axis_aligned_bounding_box()

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# GRE min and max bounds used to normalized GRE after alignment
#min_bound = np.array([-1.62603426, -2.44692874,  4.77449608])
#max_bound = np.array([ 4.453969,    2.82108521, 11.86920357])

min_bound = bbox.min_bound
max_bound = bbox.max_bound
print(min_bound, max_bound)
# sdf

normalized_points = (points - min_bound) / (max_bound - min_bound)
voxel_dimensions = 440
scaled_points = normalized_points * (voxel_dimensions - 1)

flipped_matrix = scaled_points * [1,1,1]
cropped_point_cloud = o3d.geometry.PointCloud()
cropped_point_cloud.points = o3d.utility.Vector3dVector(flipped_matrix)
cropped_point_cloud.colors = o3d.utility.Vector3dVector(pcd.colors)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_point_cloud, 1)

output_filename = r"../../data/dataset_May/RED_colmap_alignment/voxelized.ply"
o3d.io.write_voxel_grid(output_filename, voxel_grid)


pcd = o3d.io.read_point_cloud(r"../../data/dataset_May/RED_colmap_alignment/voxelized.ply")

points = np.asarray(pcd.points)

topdown_points = dict()

for point in tqdm(points):
    x, y, z = point
    key = (int(x), int(y))
    if key not in topdown_points:
        topdown_points[key] = z
    elif z > topdown_points[key]:
        topdown_points[key] = z

result_image = Image.new('RGB', (voxel_dimensions, voxel_dimensions), 0)
image_draw = ImageDraw.Draw(result_image)

image_draw.rectangle([(0, 0), (voxel_dimensions, voxel_dimensions)], fill=(0, 0, 255))

layersfolder = r"../../data/dataset_May/RED_layers_cleaned/"

image_layers = dict()
for i in range(voxel_dimensions):
  image_layers[i] = Image.open(layersfolder + f"Layer_{i+1}.png").convert('RGB')

for (x,y),z in tqdm(topdown_points.items()):
    layerimage = image_layers[int(z)]
    result_image.putpixel((x, voxel_dimensions - y - 1), layerimage.getpixel((x, voxel_dimensions - y - 1)))
    
result_image.save(r"../../data/dataset_May/RED_colmap_alignment/topdown_view.png")

pcd = o3d.io.read_point_cloud(r"../../data/dataset_May/RED_colmap_alignment/voxelized.ply")

translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.int32) # scene 2

# translation_vector = np.array([0.107006, 0.117295, -0.062955], dtype=np.int32) # scene 2
# translation_vector = np.array([30.0, -25.0, -6.0], dtype=np.int32) # scene 2
# translation_vector = np.array([-32.0, 38.0, 15.0], dtype=np.int32) # scene 3
# translation_vector = np.array([15.0, -34.0, -15.0], dtype=np.int32) # scene 4

cropped_point_cloud = o3d.geometry.PointCloud()
cropped_point_cloud.points = o3d.utility.Vector3dVector(points)
cropped_point_cloud.colors = o3d.utility.Vector3dVector(pcd.colors)
cropped_point_cloud.translate(translation_vector)
points = np.asarray(cropped_point_cloud.points)
colors = np.asarray(cropped_point_cloud.colors)
output_filename = r"../../data/dataset_May/RED_colmap_alignment/fused_translated.ply"
o3d.io.write_point_cloud(output_filename, cropped_point_cloud)
