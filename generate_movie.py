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

    viewer.save_video("trained.mp4",fps=10)



def init_dataset_generator(generate_dataset = False):
    #location of the CPSL dataset we wish to process
    dataset_folder = "/data/david/CPSL_Ground/wilkenson_datasets/"

    # scenario_folders = sorted(os.listdir(dataset_folder))
    train_scenarios = ["4th_hallway_slow","1st_hallway_slow","4th_hallway_1"]

    test_scenarios = ["1st_hallway_slow_1","1st_hallway_fast","1st_hallway_fast_spin","4th_hallway_fast","4th_hallway_fast_spin"]

    train_scenarios = [os.path.join(dataset_folder,scenario_folder) for
                    scenario_folder in train_scenarios]
    test_scenarios = [os.path.join(dataset_folder,scenario_folder) for
                    scenario_folder in [test_scenarios[2],test_scenarios[4]]]

    scenarios_to_use = test_scenarios

    #location that we wish to save the dataset to
    generated_dataset_path = "/data/david/CPSL_Ground/test/"

    #specifying the names for the files
    generated_file_name = "frame"
    radar_data_folder = "radar"
    lidar_data_folder = "lidar"

    #basic dataset settings
    num_chirps_to_save = 40
    num_previous_frames = 0

    #initialize the DatasetGenerator
    dataset_generator = DatasetGenerator()

    dataset_generator.config_generated_dataset_paths(
        generated_dataset_path=generated_dataset_path,
        generated_file_name=generated_file_name,
        generated_radar_data_folder=radar_data_folder,
        generated_lidar_data_folder=lidar_data_folder,
        clear_existing_data=False
    )

    dataset_generator.config_radar_lidar_data_paths(
        scenario_folder= scenarios_to_use[0],
        radar_data_folder=radar_data_folder,
        lidar_data_folder=lidar_data_folder
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
            lidar_data_folder=lidar_data_folder
        )
    
    return dataset_generator

if __name__ == '__main__':
    
    #change directory if needed
    # os.chdir("..")

    main()
    sys.exit()