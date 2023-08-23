import sys
import os
import numpy as np
from CPSL_Radar.Viewer import Viewer
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
    viewer = Viewer(
        dataset_generator=dataset_generator,
        transforms_to_apply= unet_transforms,
        working_dir="working_dir/CPSL_Ground",
        model_file_name="trained_focal_b1024_e100.pth",
        cuda_device='cuda:0'
    )

    viewer.save_video("trained_focal_b1024_e100.mp4",fps=10)



def init_dataset_generator(generate_dataset = False):
    #location of the CPSL dataset we wish to process
    dataset_folder = "/data/david/CPSL_Ground/recorded_datasets/"
    scenario_folders = sorted(os.listdir(dataset_folder))

    train_scenarios = [os.path.join(dataset_folder,scenario_folder) for
                   scenario_folder in scenario_folders[0:-1]]
    test_scenarios = [os.path.join(dataset_folder,scenario_folders[-1])]

    #location that we wish to save the dataset to
    generated_dataset_path = "/data/david/CPSL_Ground/test/"

    #specifying the names for the files
    generated_file_name = "frame"
    generated_radar_data_folder = "radar"
    generated_lidar_data_folder = "lidar"

    #initialize the DatasetGenerator
    dataset_generator = DatasetGenerator()

    dataset_generator.config_generated_dataset_paths(
        generated_dataset_path=generated_dataset_path,
        generated_file_name=generated_file_name,
        generated_radar_data_folder=generated_radar_data_folder,
        generated_lidar_data_folder=generated_lidar_data_folder,
        clear_existing_data=generate_dataset
    )

    dataset_generator.config_radar_lidar_data_paths(
        scenario_folders= test_scenarios,
        radar_data_folder=generated_radar_data_folder,
        lidar_data_folder=generated_lidar_data_folder
    )

    #configure the radar data processor
    dataset_generator.config_radar_data_processor(
        max_range_bin=64,
        num_chirps_to_save=40,
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
        num_angle_bins=48
    )

    if generate_dataset:
        dataset_generator.generate_dataset()
    
    return dataset_generator

if __name__ == '__main__':
    
    #change directory if needed
    # os.chdir("..")

    main()
    sys.exit()