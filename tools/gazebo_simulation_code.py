from shlex import join
import sdformat13 as sdf
import gz.math7 as gzm

from photo_shoot_config import PhotoShootConfig
from person_config import PersonConfig
from forest_config import ForestConfig
from world_config import WorldConfig
from launcher import Launcher
import numpy as np
import random
import time
import math
import os


if __name__ == "__main__":
    start_time = time.time()

    world_file_in = "../worlds/example_photo_shoot22.sdf"
    world_file_out = "../../photo_shoot.sdf"
    output_directory = "../../Desktop/test"  # "../../data/photo_shoot"
    #output_directory = "/mnt/d/CV_ExData/Third_run"  # "../../data/photo_shoot"  D:\CV_ExData\Second_run

    # Start off by loading an existing world config
    # so that we don't have to create everything!
    world_config = WorldConfig()
    world_config.load(world_file_in)
    scene = world_config.get_scene()

    ### Added person Config. (has to be done)
    
    person_config = PersonConfig()
    person_config.set_pose(gzm.Pose3d(0, 0, -1000, 0, 0, 0))
    person_config.set_model_pose("idle")
    world_config.add_plugin(person_config)    
    
    # NUM_TREES = [420, 420, 420, 420,420]
    NUM_TREES = [735, 320, 370]

    model = world_config.world.model_by_name("photo_shoot")
    link = model.link_by_name("camera_link")
    rgb_sensor = link.sensor_by_name("rgb_camera")
    rgb_camera = rgb_sensor.camera_sensor()


    far = 35.01
    near = 34.01
    MAIN_FOLDER_DIR = '/home/haitham/Desktop/New_Data'
    scene = 222222
    tree_index=0
    for i in range(2): # number of the scenes
        TREES = tree_index
        print(TREES)
        
        if i != 0 and i %2 == 0:
            scene += 1
            tree_index += 1
        # if i % 2 == 0:
        #     TREES = random.randint(0, 4)
        #     print(TREES)
        if i % 2 == 0:
        # if False:
            if not os.path.isdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1))):
                os.mkdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1)))
                if not os.path.isdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'FS')):
                    os.mkdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'FS'))


            rgb_camera.set_near_clip(1)      # Near clip plane in meters
            rgb_camera.set_far_clip(35.01)        # Far clip plane in meters 

            photo_shoot_config = PhotoShootConfig()
        

            photo_shoot_config.set_save_rgb(True) # Whether to save rgb images
            photo_shoot_config.set_save_thermal(True) # Whether to save thermal images
            photo_shoot_config.set_save_depth(True) # Whether to save depth images
            photo_shoot_config.set_depth_scaling(0.0, 2000.0)   # photo_shoot_config.set_depth_scaling(0.0, 100.0)


            ### Thermal Config. (has to be done)
            photo_shoot_config.set_direct_thermal_factor(64)  # 64    # effect the sun light
            photo_shoot_config.set_indirect_thermal_factor(5) #5      # effect the temptature of the area which is not effecting by the sun light
            photo_shoot_config.set_lower_thermal_threshold(285)    #285  # we incease we have more black spots
            photo_shoot_config.set_upper_thermal_threshold(330)   # if we incerase more gray  # decrease more white spot

            

            photo_shoot_config.set_directory(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'FS'))
            
            img_Name = f"TD"
            photo_shoot_config.set_prefix(img_Name)
            # area_size=15      
            # grid_resolution=9
            # grid_size= int((grid_resolution-1)/2)

            # for y in range( -grid_size,grid_size+1,1):
            #     for x in range(-grid_size,grid_size+1):
            #         # print(x*area_size/grid_size ,  y*area_size/grid_size )
            #         #Top Down
            #         yy = (y*area_size/grid_size) + np.random.normal(0, .5, 1)[0]
            #         xx = (-x*area_size/grid_size) + np.random.normal(0, .5, 1)[0]
            #         z = 35.01 + abs(np.random.normal(0, .5, 1)[0])
            #         with open(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), "poses.txt"), "a") as f:
            #             txt = str(xx) + "," + str(yy) + "," + str(z)
            #             f.write(txt)
            #             f.write("\n")
            #         print(yy,xx, z)
            #         photo_shoot_config.add_poses([gzm.Pose3d(yy, xx, z , 0.0, 1.57079632679, 0)]) 
            #         # photo_shoot_config.add_poses([gzm.Pose3d(0, 0, 17 , 0.0, 1.57079632679, -90)]) 
            

            area_size = 24
            grid_resolution = 9

            grid_size= int((grid_resolution-1)/2)

            spacing = area_size / (grid_resolution - 1)


            for i in range( -grid_size,grid_size+1,1):
                for j in range(-grid_size,grid_size+1):
                    x = i * spacing
                    y = j * spacing
                    with open(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), "poses.txt"), "a") as f:
                        txt = str(y) + "," + str(x) + "," + str(35)
                        f.write(txt)
                        f.write("\n")
                    print(x,y, 35)
                    photo_shoot_config.add_poses([gzm.Pose3d(x, y, 35 , 0.0, 1.57079632679, 0)]) 
   

            world_config.add_plugin(photo_shoot_config)

            forest_config = ForestConfig()
            forest_config.set_generate(True)
            forest_config.set_ground_texture(7)  # Have a look at the respective model for options
            forest_config.set_direct_spawning(True)  # Does not work when using the gazebo gui, but 3x faster
            forest_config.set_ground_temperature(288.15)
            # forest_config.set_ground_thermal_texture(
            #     0,              # Texture index (From the model)
            #     288.15,         # Minimal temperature in Kelvin
            #     320.0           # Maximal temperature in Kelvin
            # )
            forest_config.set_trunk_temperature(290) # 291.15   # In Kelvin
            forest_config.set_twigs_temperature(287.15)  # In Kelvin     #287.15
            forest_config.set_size(100)  # Width and height of the forest

            
            # forest_config.set_trees(400)  # Number of treees
            forest_config.set_trees(NUM_TREES[TREES])  # Number of treees  NUM_TREES[0]

            forest_config.set_seed(scene)       # Change the seed for multiple runs!
            forest_config.set_species("Birch", {
                        "percentage": 1.0,  # Percentage that this species makes up of all trees
                        "homogeneity": 0.95,
                        "trunk_texture": 0,  # Have a look at the respective model for options
                        "twigs_texture": 0,  # Have a look at the respective model for options
                        "tree_properties": {
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
                            "twig_scale": 0.2}})



            world_config.add_plugin(forest_config)

            # Save the modified config
            world_config.save(world_file_out)
            # Launch the simulation
            launcher = Launcher()
            launcher.set_launch_config("server_only", True)
            launcher.set_launch_config("running", True)
            launcher.set_launch_config("iterations", 2)
            launcher.set_launch_config("world", world_file_out)
            print(launcher.launch())

        else:

            if not os.path.isdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'ZS')):
                os.mkdir(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'ZS'))

            far = 35.05
            near = 34.78            
            #far = 21.88
            #near = 21.61
            kk = 0.03
            rgb_camera.set_near_clip(5)      # Near clip plane in meters

            for j in range(550):
                if j % 1 == 0:

                    print(j)
                    # rgb_camera.set_near_clip(near)      # Near clip plane in meters
                    # rgb_camera.set_far_clip(far)        # Far clip plane in meters 
                    # if j != 0:
                    #     rgb_camera.set_near_clip(near+0.01)      # Near clip plane in meters
                    #     rgb_camera.set_far_clip(far-0.01)        # Far clip plane in meters 
                    #     # rgb_camera.set_near_clip(near+0.48)      # Near clip plane in meters
                    #     # rgb_camera.set_far_clip(far-0.498 - 0.002) 
                    # else:
                    #     rgb_camera.set_near_clip(near+0.98)      # Near clip plane in meters
                    #     rgb_camera.set_far_clip(far)        # Far clip plane in meters 
                    
                    # rgb_camera.set_near_clip(near)      # Near clip plane in meters
                    rgb_camera.set_far_clip(far)        # Far clip plane in meters 
                    rgb_camera.set_near_clip(near)        # Far clip plane in meters 
                    # Far clip plane in meters 

                    photo_shoot_config = PhotoShootConfig()
            
                    photo_shoot_config.set_save_rgb(True) # Whether to save rgb images
                    photo_shoot_config.set_save_thermal(False) # Whether to save thermal images
                    photo_shoot_config.set_save_depth(False) # Whether to save depth images
                    photo_shoot_config.set_depth_scaling(0.0, 2000.0)   # photo_shoot_config.set_depth_scaling(0.0, 100.0)
                    
                
                    photo_shoot_config.set_directory(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'ZS'))

                    img_Name = f"TD"

                    photo_shoot_config.set_prefix(img_Name+str(j)) 


                    # Y, X, Z first three parameter in Pose3d
                    photo_shoot_config.add_poses([gzm.Pose3d(0.0, 0.0, 35.0, 0.0, 1.57079632679, 0)]) 

                    # if j != 0:
                    near -= 0.03
                    far -= 0.03

                    world_config.add_plugin(photo_shoot_config)

                    forest_config = ForestConfig()
                    forest_config.set_generate(True)
                    forest_config.set_ground_texture(7)  # Have a look at the respective model for options
                    forest_config.set_direct_spawning(True)  # Does not work when using the gazebo gui, but 3x faster
                    forest_config.set_ground_temperature(288.15)  # In Kelvin    #288.15
                    forest_config.set_trunk_temperature(290) # 291.15   # In Kelvin
                    forest_config.set_twigs_temperature(287.15)  # In Kelvin     #287.15
                    forest_config.set_size(100)  # Width and height of the forest

                    
                    forest_config.set_trees(NUM_TREES[TREES])  # Number of treees  NUM_TREES[0]

                    forest_config.set_seed(scene)       # Change the seed for multiple runs!
                    forest_config.set_species("Birch", {
                                "percentage": 1.0,  # Percentage that this species makes up of all trees
                                "homogeneity": 0.95,
                                "trunk_texture": 0,  # Have a look at the respective model for options
                                "twigs_texture": 0,  # Have a look at the respective model for options
                                "tree_properties": {
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
                                    "twig_scale": 0.2}})



                    world_config.add_plugin(forest_config)

                    # Save the modified config
                    world_config.save(world_file_out)
                    # Launch the simulation
                    launcher = Launcher()
                    launcher.set_launch_config("server_only", True)
                    launcher.set_launch_config("running", True)
                    launcher.set_launch_config("iterations", 2)
                    launcher.set_launch_config("world", world_file_out)
                    print(launcher.launch())
                # else:

                #     print(j)
                #     # rgb_camera.set_near_clip(near)      # Near clip plane in meters
                #     # rgb_camera.set_far_clip(far)        # Far clip plane in meters 
                #     # if j != 0:
                #     #     rgb_camera.set_near_clip(near+0.01)      # Near clip plane in meters
                #     #     rgb_camera.set_far_clip(far-0.01)        # Far clip plane in meters 
                #     #     # rgb_camera.set_near_clip(near+0.48)      # Near clip plane in meters
                #     #     # rgb_camera.set_far_clip(far-0.498 - 0.002) 
                #     # else:
                #     #     rgb_camera.set_near_clip(near+0.98)      # Near clip plane in meters
                #     #     rgb_camera.set_far_clip(far)        # Far clip plane in meters 
                    
                #     # rgb_camera.set_near_clip(near)      # Near clip plane in meters
                #     rgb_camera.set_far_clip(far + 0.13)        # Far clip plane in meters 
                #     # Far clip plane in meters 

                #     photo_shoot_config = PhotoShootConfig()
            
                #     photo_shoot_config.set_save_rgb(True) # Whether to save rgb images
                #     photo_shoot_config.set_save_thermal(False) # Whether to save thermal images
                #     photo_shoot_config.set_save_depth(False) # Whether to save depth images
                #     photo_shoot_config.set_depth_scaling(0.0, 2000.0)   # photo_shoot_config.set_depth_scaling(0.0, 100.0)
                    
                
                #     photo_shoot_config.set_directory(os.path.join(MAIN_FOLDER_DIR, 'Scene_'+str(scene+1), 'ZS'))

                #     img_Name = f"TD"

                #     photo_shoot_config.set_prefix(img_Name+str(j)) 



                #     photo_shoot_config.add_poses([gzm.Pose3d(0, 0, 35, 0.0, 1.57079632679, 0)]) 

                #     # if j != 0:
                #     near -= 0.27
                #     far -= 0.27

                #     world_config.add_plugin(photo_shoot_config)

                #     forest_config = ForestConfig()
                #     forest_config.set_generate(True)
                #     forest_config.set_ground_texture(7)  # Have a look at the respective model for options
                #     forest_config.set_direct_spawning(True)  # Does not work when using the gazebo gui, but 3x faster
                #     forest_config.set_ground_temperature(288.15)  # In Kelvin    #288.15
                #     forest_config.set_trunk_temperature(290) # 291.15   # In Kelvin
                #     forest_config.set_twigs_temperature(287.15)  # In Kelvin     #287.15
                #     forest_config.set_size(100)  # Width and height of the forest

                    
                #     forest_config.set_trees(NUM_TREES[TREES])  # Number of treees  NUM_TREES[0]

                #     forest_config.set_seed(scene)       # Change the seed for multiple runs!
                #     forest_config.set_species("Birch", {
                #                 "percentage": 1.0,  # Percentage that this species makes up of all trees
                #                 "homogeneity": 0.95,
                #                 "trunk_texture": 0,  # Have a look at the respective model for options
                #                 "twigs_texture": 0,  # Have a look at the respective model for options
                #                 "tree_properties": {
                #                     "clump_max": 0.45,
                #                     "clump_min": 0.4,
                #                     "length_falloff_factor": 0.65,
                #                     "length_falloff_power": 0.75,
                #                     "branch_factor": 2.45,
                #                     "radius_falloff_rate": 0.7,
                #                     "climb_rate": 0.55,
                #                     "taper_rate": 0.8,
                #                     "twist_rate": 8.0,
                #                     "segments": 6,
                #                     "levels": 6,
                #                     "sweep_amount": 0.0,
                #                     "initial_branch_length": 0.7,
                #                     "trunk_length": 0.5,
                #                     "drop_amount": 0.0,
                #                     "grow_amount": 0.4,
                #                     "v_multiplier": 0.2,
                #                     "twig_scale": 0.2}})



                #     world_config.add_plugin(forest_config)

                #     # Save the modified config
                #     world_config.save(world_file_out)
                #     # Launch the simulation
                #     launcher = Launcher()
                #     launcher.set_launch_config("server_only", True)
                #     launcher.set_launch_config("running", True)
                #     launcher.set_launch_config("iterations", 2)
                #     launcher.set_launch_config("world", world_file_out)
                #     print(launcher.launch())

                # End of scene
    
    print(f"Time taken = {time.time() - start_time}")





