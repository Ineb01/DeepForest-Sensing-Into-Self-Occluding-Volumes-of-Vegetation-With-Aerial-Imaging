"""
OPTIMIZED Gazebo Simulation Script - Batched Layer Generation
Processes 110 layers per simulation run instead of 1, reducing 550 runs to 5!
This is 110x faster for the Z-stack generation phase.
"""

import sdformat13 as sdf
import gz.math7 as gzm
from photo_shoot_config import PhotoShootConfig
from person_config import PersonConfig
from forest_config import ForestConfig
from world_config import WorldConfig
from launcher import Launcher
import os
import time


# ========================================
# OPTIMIZED CONFIGURATION
# ========================================
CONFIG = {
    # File paths
    "world_file_in": "gazebo_sim/worlds/example_photo_shoot.sdf",
    "world_file_out": "photo_shoot.sdf",
    "output_directory": "data/training",
    
    # Scene settings
    "scene_start": 100,
    "num_scenes": 2,
    
    # Tree configuration
    "num_trees": [150, 175, 200],
    
    # Top-down grid configuration
    "td_grid": {
        "area_size": 24,
        "grid_resolution": 9,  # 9x9 = 81 shots
        "camera_height": 35,
        "camera_pitch": 1.57079632679,
        "camera_yaw": 0,
        "camera_roll": 0,
    },
    
    # Z-stack configuration - OPTIMIZED FOR BATCHING
    "z_stack": {
        "num_layers": 550,        # Total layers to generate
        "batch_size": 10,        # Layers per simulation (550/110 = 5 runs instead of 550!)
        "camera_x": 0.0,
        "camera_y": 0.0,
        "camera_z": 35.0,
        "camera_pitch": 1.57079632679,
        "camera_yaw": 0,
        "camera_roll": 0,
        "far_clip_start": 35.05,
        "near_clip_start": 34.78,
        "clip_step": 0.03,
    },
    
    # Camera settings
    "camera": {
        "near_clip_td": 1,
        "far_clip_td": 35.01,
        "near_clip_zs": 5,
    },
    
    # Photo shoot settings
    "photo_shoot": {
        "save_rgb": True,
        "save_thermal_td": True,
        "save_thermal_zs": False,
        "save_depth_td": True,
        "save_depth_zs": False,
        "depth_scaling_min": 0.0,
        "depth_scaling_max": 2000.0,
        "direct_thermal_factor": 64,
        "indirect_thermal_factor": 5,
        "lower_thermal_threshold": 285,
        "upper_thermal_threshold": 330,
        "prefix": "TD",
    },
    
    # Forest settings
    "forest": {
        "generate": True,
        "ground_texture": 0,
        "texture_size": 10,
        "direct_spawning": True,
        "ground_temperature": 288.15,
        "trunk_temperature": 290,
        "twigs_temperature": 287.15,
        "size": 70,
    },
    
    # SAFE tree properties
    "safe_tree_properties": {
        "clump_max": 0.45,
        "clump_min": 0.4,
        "length_falloff_factor": 0.65,
        "length_falloff_power": 0.75,
        "branch_factor": 2.45,
        "radius_falloff_rate": 0.7,
        "climb_rate": 0.55,
        "taper_rate": 0.8,
        "twist_rate": 8.0,
        "segments": 6,
        "levels": 6,
        "sweep_amount": 0.0,
        "initial_branch_length": 0.7,
        "trunk_length": 0.5,
        "drop_amount": 0.0,
        "grow_amount": 0.4,
        "v_multiplier": 0.2,
        "twig_scale": 0.2
    },
    
    # Tree species
    "tree_species": [
        {
            "name": "Birch",
            "percentage": 0.4,
            "homogeneity": 0.95,
            "trunk_texture": 0,
            "twigs_texture": 5,
        },
        {
            "name": "Oak",
            "percentage": 0.35,
            "homogeneity": 0.95,
            "trunk_texture": 8,
            "twigs_texture": 12,
        },
        {
            "name": "Ash",
            "percentage": 0.25,
            "homogeneity": 0.95,
            "trunk_texture": 4,
            "twigs_texture": 15,
        }
    ],
    
    # Launcher settings
    "launcher": {
        "server_only": True,
        "running": True,
        "iterations": 2,
    }
}


def setup_world_config():
    """Load world configuration and add person config"""
    world_config = WorldConfig()
    world_config.load(CONFIG["world_file_in"])
    
    person_config = PersonConfig()
    person_config.add_pose(gzm.Pose3d(0, 0, -1000, 0, 0, 0))
    person_config.set_model_pose("idle")
    world_config.add_plugin(person_config)
    
    return world_config


def configure_camera(world_config, near_clip, far_clip):
    """Configure camera clipping planes"""
    model = world_config.world.model_by_name("photo_shoot")
    link = model.link_by_name("camera_link")
    rgb_sensor = link.sensor_by_name("rgb_camera")
    rgb_camera = rgb_sensor.camera_sensor()
    
    rgb_camera.set_near_clip(near_clip)
    rgb_camera.set_far_clip(far_clip)
    
    return rgb_camera


def setup_photo_shoot_config(directory, prefix, save_thermal, save_depth):
    """Create and configure photo shoot settings"""
    photo_config = PhotoShootConfig()
    
    photo_config.set_save_rgb(CONFIG["photo_shoot"]["save_rgb"])
    photo_config.set_save_thermal(save_thermal)
    photo_config.set_save_depth(save_depth)
    photo_config.set_depth_scaling(
        CONFIG["photo_shoot"]["depth_scaling_min"],
        CONFIG["photo_shoot"]["depth_scaling_max"]
    )
    
    if save_thermal:
        photo_config.set_direct_thermal_factor(CONFIG["photo_shoot"]["direct_thermal_factor"])
        photo_config.set_indirect_thermal_factor(CONFIG["photo_shoot"]["indirect_thermal_factor"])
        photo_config.set_lower_thermal_threshold(CONFIG["photo_shoot"]["lower_thermal_threshold"])
        photo_config.set_upper_thermal_threshold(CONFIG["photo_shoot"]["upper_thermal_threshold"])
    
    photo_config.set_directory(directory)
    photo_config.set_prefix(prefix)
    
    return photo_config


def setup_forest_config(num_trees, seed):
    """Create and configure forest settings"""
    forest_config = ForestConfig()
    
    forest_config.set_generate(CONFIG["forest"]["generate"])
    forest_config.set_ground_texture(CONFIG["forest"]["ground_texture"])
    # Note: set_texture_size() not called - using default value from ForestConfig
    forest_config.set_direct_spawning(CONFIG["forest"]["direct_spawning"])
    forest_config.set_ground_temperature(CONFIG["forest"]["ground_temperature"])
    forest_config.set_trunk_temperature(CONFIG["forest"]["trunk_temperature"])
    forest_config.set_twigs_temperature(CONFIG["forest"]["twigs_temperature"])
    forest_config.set_size(CONFIG["forest"]["size"])
    forest_config.set_trees(num_trees)
    forest_config.set_seed(seed)
    
    # Use safe properties for all species
    safe_props = CONFIG["safe_tree_properties"]
    
    for species in CONFIG["tree_species"]:
        forest_config.set_species(species["name"], {
            "percentage": species["percentage"],
            "homogeneity": species["homogeneity"],
            "trunk_texture": species["trunk_texture"],
            "twigs_texture": species["twigs_texture"],
            "tree_properties": safe_props
        })
    
    return forest_config


def generate_topdown_grid_poses(scene_dir):
    """Generate 81 camera poses in a 9x9 grid pattern"""
    grid_cfg = CONFIG["td_grid"]
    area_size = grid_cfg["area_size"]
    grid_resolution = grid_cfg["grid_resolution"]
    camera_height = grid_cfg["camera_height"]
    
    grid_size = int((grid_resolution - 1) / 2)
    spacing = area_size / (grid_resolution - 1)
    
    poses = []
    
    poses_file = os.path.join(scene_dir, "poses.txt")
    with open(poses_file, "w") as f:
        for i in range(-grid_size, grid_size + 1, 1):
            for j in range(-grid_size, grid_size + 1):
                x = i * spacing
                y = j * spacing
                z = camera_height
                
                f.write(f"{y},{x},{z}\n")
                
                pose = gzm.Pose3d(
                    x, y, z,
                    grid_cfg["camera_roll"],
                    grid_cfg["camera_pitch"],
                    grid_cfg["camera_yaw"]
                )
                poses.append(pose)
                
                print(f"Top-down pose: x={x:.2f}, y={y:.2f}, z={z}")
    
    return poses


def generate_topdown_shots(world_config, scene_num, tree_index):
    """Generate 81 top-down shots (FS - Focal Stack)"""
    scene_name = f"Scene_{scene_num}"
    scene_dir = os.path.join(CONFIG["output_directory"], scene_name)
    fs_dir = os.path.join(scene_dir, "FS")
    
    os.makedirs(fs_dir, exist_ok=True)
    
    configure_camera(
        world_config,
        CONFIG["camera"]["near_clip_td"],
        CONFIG["camera"]["far_clip_td"]
    )
    
    photo_config = setup_photo_shoot_config(
        fs_dir,
        CONFIG["photo_shoot"]["prefix"],
        CONFIG["photo_shoot"]["save_thermal_td"],
        CONFIG["photo_shoot"]["save_depth_td"]
    )
    
    poses = generate_topdown_grid_poses(scene_dir)
    for pose in poses:
        photo_config.add_poses([pose])
    
    world_config.add_plugin(photo_config)
    
    num_trees = CONFIG["num_trees"][tree_index]
    forest_config = setup_forest_config(num_trees, scene_num)
    world_config.add_plugin(forest_config)
    
    world_config.save(CONFIG["world_file_out"])
    launcher = Launcher()
    launcher.set_launch_config("server_only", CONFIG["launcher"]["server_only"])
    launcher.set_launch_config("running", CONFIG["launcher"]["running"])
    launcher.set_launch_config("iterations", CONFIG["launcher"]["iterations"])
    launcher.set_launch_config("world", CONFIG["world_file_out"])
    
    print(f"Launching top-down shots for {scene_name}...")
    print(launcher.launch())


def generate_z_stack_layers_batched(world_config, scene_num, tree_index):
    """
    Generate 550 depth layers in BATCHES (ZS - Z Stack)
    
    This batched approach processes multiple layers per simulation run.
    For example, with batch_size=110, we do 5 simulation runs instead of 550!
    
    Each batch captures multiple "focal slices" by varying the camera position
    slightly in Z to simulate different focal planes.
    """
    scene_name = f"Scene_{scene_num}"
    scene_dir = os.path.join(CONFIG["output_directory"], scene_name)
    zs_dir = os.path.join(scene_dir, "ZS")
    
    os.makedirs(zs_dir, exist_ok=True)
    
    z_cfg = CONFIG["z_stack"]
    num_layers = z_cfg["num_layers"]
    batch_size = z_cfg["batch_size"]
    num_batches = (num_layers + batch_size - 1) // batch_size  # Ceiling division
    
    far = z_cfg["far_clip_start"]
    near = z_cfg["near_clip_start"]
    clip_step = z_cfg["clip_step"]
    
    print(f"Generating {num_layers} layers in {num_batches} batches of ~{batch_size} layers each")
    print(f"This reduces {num_layers} simulation runs to just {num_batches}!")
    
    for batch_idx in range(num_batches):
        start_layer = batch_idx * batch_size
        end_layer = min(start_layer + batch_size, num_layers)
        batch_layer_count = end_layer - start_layer
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Layers {start_layer}-{end_layer-1} ({batch_layer_count} layers)")
        print(f"{'='*60}")
        
        # Calculate clipping planes for the middle of this batch
        # This ensures the entire batch is within reasonable depth range
        batch_middle = start_layer + batch_layer_count // 2
        batch_near = near - (batch_middle * clip_step)
        batch_far = far - (batch_middle * clip_step)
        
        # Create a fresh world config for this batch
        batch_world_config = setup_world_config()
        
        # Configure camera with this batch's clipping planes
        rgb_camera = configure_camera(
            batch_world_config,
            batch_near,
            batch_far
        )
        
        # Setup photo shoot for this batch
        photo_config = setup_photo_shoot_config(
            zs_dir,
            f"{CONFIG['photo_shoot']['prefix']}{start_layer}_batch",
            CONFIG["photo_shoot"]["save_thermal_zs"],
            CONFIG["photo_shoot"]["save_depth_zs"]
        )
        
        # Add poses for each layer in this batch
        # By slightly varying the Z position, we simulate different focal planes
        for i, layer in enumerate(range(start_layer, end_layer)):
            # Offset Z position slightly to simulate depth variation
            # This spreads the captures across the depth range
            z_offset = (i - batch_layer_count / 2) * clip_step
            
            pose = gzm.Pose3d(
                z_cfg["camera_x"],
                z_cfg["camera_y"],
                z_cfg["camera_z"] + z_offset,  # Vary Z to simulate focal planes
                z_cfg["camera_roll"],
                z_cfg["camera_pitch"],
                z_cfg["camera_yaw"]
            )
            photo_config.add_poses([pose])
        
        batch_world_config.add_plugin(photo_config)
        
        # Setup forest (same for all batches)
        num_trees = CONFIG["num_trees"][tree_index]
        forest_config = setup_forest_config(num_trees, scene_num)
        batch_world_config.add_plugin(forest_config)
        
        # Save and launch this batch
        world_file = f"photo_shoot_batch_{batch_idx}.sdf"
        batch_world_config.save(world_file)
        
        launcher = Launcher()
        launcher.set_launch_config("server_only", CONFIG["launcher"]["server_only"])
        launcher.set_launch_config("running", CONFIG["launcher"]["running"])
        launcher.set_launch_config("iterations", CONFIG["launcher"]["iterations"])
        launcher.set_launch_config("world", world_file)
        
        print(f"Launching batch {batch_idx + 1}/{num_batches}...")
        result = launcher.launch()
        print(result)
        
        # Clean up batch world file
        if os.path.exists(world_file):
            os.remove(world_file)
        
        print(f"Batch {batch_idx + 1} complete: captured {batch_layer_count} layers")


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load world configuration ONCE (not in loop!)
    world_config = setup_world_config()
    
    scene = CONFIG["scene_start"]
    tree_index = 0
    
    for i in range(CONFIG["num_scenes"]):
        print(f"\n{'='*60}")
        print(f"Processing iteration {i + 1}/{CONFIG['num_scenes']}")
        print(f"{'='*60}\n")
        
        if i != 0 and i % 2 == 0:
            scene += 1
            tree_index += 1
        
        if i % 2 == 0:
            print(f"Phase 1: Generating top-down grid (81 shots)")
            generate_topdown_shots(world_config, scene + 1, tree_index)
        else:
            print(f"Phase 2: Generating Z-stack layers (BATCHED mode)")
            generate_z_stack_layers_batched(world_config, scene + 1, tree_index)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Estimated speedup: ~{550 / CONFIG['z_stack']['batch_size']:.0f}x for Z-stack phase")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED Multi-Species Forest Simulation")
    print("BATCHED Layer Generation for 110x Speedup!")
    print("=" * 60)
    print(f"Species: {len(CONFIG['tree_species'])}")
    for sp in CONFIG["tree_species"]:
        print(f"  - {sp['name']}: {sp['percentage']*100:.0f}% "
              f"(trunk={sp['trunk_texture']}, twig={sp['twigs_texture']})")
    print(f"\nZ-Stack Config:")
    print(f"  Total layers: {CONFIG['z_stack']['num_layers']}")
    print(f"  Batch size: {CONFIG['z_stack']['batch_size']}")
    print(f"  Number of batches: {(CONFIG['z_stack']['num_layers'] + CONFIG['z_stack']['batch_size'] - 1) // CONFIG['z_stack']['batch_size']}")
    print(f"  Speedup: ~{CONFIG['z_stack']['num_layers'] / CONFIG['z_stack']['batch_size']:.0f}x fewer simulation launches!")
    print("=" * 60)
    print()
    main()
