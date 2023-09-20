import numpy as np
from multiprocessing.connection import Client
import sys

from CPSL_Radar.Analyzer import Analyzer
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator
from CPSL_Radar.models.unet import unet
from torchvision import transforms

class RealTimeModel():

    def __init__(self):
        
        #dataste generator and analyzer for real time model operation
        self.analyzer:Analyzer = None

        self.conn:Client = None

        dataset_generator = self._init_dataset_generator()

        self.init_model(dataset_generator)

        self._connect_to_radar()

        self.run()

    def run(self):

        while True:

            try:
                #receive the range azimuth response
                norm_rng_az_resp = self.conn.recv()

                #send out the computed cartesian point cloud
                points_cartesian = self.analyzer.compute_predicted_point_cloud_cartesian(norm_rng_az_resp)

                self.conn.send(points_cartesian)
            
            except EOFError:
                print("RealTimeModel.run: connection closed by Radar")
                break
    
    def _connect_to_radar(self):

        print("Connecting to listener")
        address = ('localhost', 6001)     # family is deduced to be 'AF_INET'
        authkey_str = "DCA1000_client"
        self.conn = Client(address, authkey=authkey_str.encode())
        print("Connected to listener")
    
    def _init_dataset_generator(self):
        # #initialize the dataset generator
        # #campus dataset folders
        # campus_folder = "/home/david/CPSL_Ground/campus_test_dataset/"
        # #campus_folder = "/home/locobot/data/campus_datasets/"

        # campus_scenarios = ["scene_{}".format(i+1) for i in range(3)]
        # campus_test_scenarios = ["scene_{}_test".format(i+1) for i in range(7)]
        # campus_test_scenarios_spin = ["scene_{}_test_spin".format(i) for i in range(3,5)]

        # train_scenarios = [os.path.join(campus_folder,scenario_folder) for 
        #                 scenario_folder in campus_scenarios]

        # test_scenarios = [os.path.join(campus_folder,scenario_folder) for 
        #                 scenario_folder in campus_test_scenarios]

        # test_scenarios_spin = [os.path.join(campus_folder,scenario_folder) for 
        #                 scenario_folder in campus_test_scenarios_spin]
        # # wilkenson dataset
        # # wilkenson_folder = "/data/david/CPSL_Ground/wilkenson_datasets"
        # # wilkenson_scenarios = ["scene_{}".format(i+1) for i in range(10)]

        # # wilkenson_test_scenarios = ["scene_{}_test".format(i+1) for i in range(10)]

        # # train_scenarios.extend([os.path.join(wilkenson_folder,scenario_folder) for
        # #                    scenario_folder in wilkenson_scenarios])
        # # test_scenarios.extend([os.path.join(wilkenson_folder,scenario_folder) for
        # #                    scenario_folder in wilkenson_test_scenarios])

        # #box_dataset_folders
        # # box_folder = "/data/david/CPSL_Ground/box_datasets/"

        # # box_scenarios = ["scene_{}".format(i+1) for i in range(5)]

        # # train_scenarios.extend(
        # #     [os.path.join(box_folder,scenario_folder) for
        # #      scenario_folder in box_scenarios[0:-1]]
        # # )

        # # test_scenarios.extend([
        # #     os.path.join(box_folder,scenario_folder) for 
        # #     scenario_folder in box_scenarios[-1]])

        # scenarios_to_use = train_scenarios

        # #location that we wish to save the dataset to
        # generated_dataset_path = "/home/david/CPSL_Ground/test/"
        # #generated_dataset_path = "/home/locobot/data/test/"

        # #specifying the names for the files
        # generated_file_name = "frame"
        # radar_data_folder = "radar"
        # lidar_data_folder = "lidar"

        #basic dataset settings
        num_chirps_to_save = 40
        num_previous_frames = 0
        use_average_range_az = False

        #initialize the DatasetGenerator
        dataset_generator = DatasetGenerator()

        # dataset_generator.config_generated_dataset_paths(
        #     generated_dataset_path=generated_dataset_path,
        #     generated_file_name=generated_file_name,
        #     generated_radar_data_folder=radar_data_folder,
        #     generated_lidar_data_folder=lidar_data_folder,
        #     clear_existing_data=False
        # )

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

        return dataset_generator
    
    def init_model(self,dataset_generator):

        #initialize the transforms
        unet_transforms = [
            transforms.ToTensor(),
            transforms.Resize((64,48))
        ]

        #initialize the unet
        unet_model = unet(
            encoder_input_channels= 40,
            encoder_out_channels= (64,128,256),
            decoder_input_channels= (512,256,128),
            decoder_out_channels= 64,
            output_channels= 1,
            retain_dimmension= False,
            input_dimmensions= (64,48)
        )

        #initialize the viewer
        self.analyzer = Analyzer(
            dataset_generator=dataset_generator,
            model=unet_model,
            transforms_to_apply= unet_transforms,
            working_dir="working_dir/",
            model_state_dict_file_name="trained_campus_chirps_smaller.pth",
            cuda_device="cuda:0"
        )

        return
    
def main():

    RealTimeModel()

    return

#create the controller object
if __name__ == '__main__':

    main()
    sys.exit()