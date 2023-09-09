import sys
import os
import numpy as np
from CPSL_Radar.Analyzer import Analyzer
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator
from torchvision import transforms

def main():
    
    dataset_generator = init_dataset_generator(generate_dataset=False)

    #initialize the transforms
    input_transforms = [
        transforms.ToTensor(),
        transforms.Resize((64,48))
    ]

   #initialize the viewer
    viewer = Analyzer(
        dataset_generator=dataset_generator,
        transforms_to_apply= input_transforms,
        working_dir="working_dir/",
        model_file_name="trained.pth",
        cuda_device="cuda:0"
    )

    viewer.save_video("trained_adding_noise.mp4",fps=10)



def init_dataset_generator(generate_dataset = False):
    #campus dataset folders
    campus_folder = "/data/david/CPSL_Ground/campus_datasets/"

    campus_scenarios = ["scene_{}".format(i+1) for i in range(7)]
    campus_test_scenarios = ["scene_{}_test".format(i+1) for i in range(7)]
    campus_test_scenarios_spin = ["scene_{}_test_spin".format(i) for i in range(3,5)]

    train_scenarios = [os.path.join(campus_folder,scenario_folder) for 
                    scenario_folder in campus_scenarios]

    test_scenarios = [os.path.join(campus_folder,scenario_folder) for 
                    scenario_folder in campus_test_scenarios]

    test_scenarios_spin = [os.path.join(campus_folder,scenario_folder) for 
                    scenario_folder in campus_test_scenarios_spin]
    # wilkenson dataset
    # wilkenson_folder = "/data/david/CPSL_Ground/wilkenson_datasets"
    # wilkenson_scenarios = ["scene_{}".format(i+1) for i in range(10)]

    # wilkenson_test_scenarios = ["scene_{}_test".format(i+1) for i in range(10)]

    # train_scenarios.extend([os.path.join(wilkenson_folder,scenario_folder) for
    #                    scenario_folder in wilkenson_scenarios])
    # test_scenarios.extend([os.path.join(wilkenson_folder,scenario_folder) for
    #                    scenario_folder in wilkenson_test_scenarios])

    #box_dataset_folders
    # box_folder = "/data/david/CPSL_Ground/box_datasets/"

    # box_scenarios = ["scene_{}".format(i+1) for i in range(5)]

    # train_scenarios.extend(
    #     [os.path.join(box_folder,scenario_folder) for
    #      scenario_folder in box_scenarios[0:-1]]
    # )

    # test_scenarios.extend([
    #     os.path.join(box_folder,scenario_folder) for 
    #     scenario_folder in box_scenarios[-1]])

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
    use_average_range_az= True
    

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
        use_average_range_az= use_average_range_az,
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