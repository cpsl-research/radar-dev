import sys
import os
import numpy as np
from CPSL_Radar.Analyzer import Analyzer
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator
from torchvision import transforms

def main():
    
    dataset_generator = init_dataset_generator(generate_dataset=False)

    #initialize the transforms
    unet_transforms = [
        transforms.ToTensor(),
        transforms.Resize((64,48))
    ]

   #initialize the viewer
    viewer = Analyzer(
        dataset_generator=dataset_generator,
        transforms_to_apply= unet_transforms,
        working_dir="working_dir/",
        model_file_name="trained.pth",
        cuda_device="cuda:0"
    )

    viewer.save_video("drone_test.mp4",fps=10)



def init_dataset_generator(generate_dataset = False):
    #initialize the dataset generator
    drone_folder = "/data/david/CPSL_Drone/"
    test_scenarios = ["drone_test"]

    test_scenarios = [os.path.join(drone_folder,scenario_folder) for
                    scenario_folder in test_scenarios]

    scenarios_to_use = test_scenarios

    #location that we wish to save the dataset to
    generated_dataset_path = "/data/david/CPSL_Drone/test/"

    #specifying the names for the files
    generated_file_name = "frame"
    radar_data_folder = "radar"

    #basic dataset settings
    num_chirps_to_save = 40
    num_previous_frames = 0

    #initialize the DatasetGenerator
    dataset_generator = DatasetGenerator(radar_data_only=True)

    dataset_generator.config_generated_dataset_paths(
        generated_dataset_path=generated_dataset_path,
        generated_file_name=generated_file_name,
        generated_radar_data_folder=radar_data_folder,
        generated_lidar_data_folder=None,
        clear_existing_data=False
    )

    dataset_generator.config_radar_lidar_data_paths(
        scenario_folder= scenarios_to_use[0],
        radar_data_folder=radar_data_folder,
        lidar_data_folder=None
    )

    #configure the radar data processor
    dataset_generator.config_radar_data_processor(
        max_range_bin=64,
        num_chirps_to_save=num_chirps_to_save,
        num_previous_frames=num_previous_frames,
        radar_fov= [-0.87, 0.87], #+/- 50 degrees
        num_angle_bins=64,
        power_range_dB=[60,105],
        chirps_per_frame= 64,
        rx_channels = 4,
        tx_channels = 1,
        samples_per_chirp = 64,
        adc_sample_rate_Hz = 2e6,
        chirp_slope_MHz_us= 35,
        start_freq_Hz=77e9,
        idle_time_us = 100,
        ramp_end_time_us = 100
    )

    #configure the lidar data processor
    dataset_generator.config_lidar_data_processor(
        max_range_m=8.56,
        num_range_bins=64,
        angle_range_rad=[-np.pi/2 - 0.87,-np.pi/2 + 0.87], #[-np.pi /2 , np.pi /2],
        num_angle_bins=48,
        num_previous_frames=num_previous_frames
    )

    if generate_dataset:
        dataset_generator.generate_dataset_from_multiple_scenarios(
            scenario_folders = scenarios_to_use,
            radar_data_folder= radar_data_folder,
            lidar_data_folder=None
        )
    
    return dataset_generator

if __name__ == '__main__':
    
    #change directory if needed
    # os.chdir("..")

    main()
    sys.exit()