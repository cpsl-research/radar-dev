#data processors
from CPSL_Radar.data_processors.Radar_Data_Processor import RadarDataProcessor
from CPSL_Radar.data_processors.Lidar_Data_Processor import LidarDataProcessor

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

class DatasetGenerator:

    def __init__(self):

        #parameters for the generated dataset
        self.generated_dataset_path:str = None
        self.generated_file_name:str = None
        self.generated_radar_data_folder:str = None
        self.generated_lidar_data_foler:str = None
        self.generated_dataset_paths_set = False

        #parameters for the data used to generate the dataset
        self.radar_data_paths:list = None
        self.lidar_data_paths:list = None
        self.lidar_radar_data_paths_set = False

        #dataset processors
        self.radar_data_processor = RadarDataProcessor()
        self.lidar_data_processor = LidarDataProcessor()
        
        self.num_samples = None

    def config_generated_dataset_paths(
            self,
            generated_dataset_path:str,
            generated_file_name:str = "frame",
            generated_radar_data_folder:str = "radar",
            generated_lidar_data_folder:str = "lidar",
            clear_existing_data:bool = False
    ):
        
        self.generated_dataset_path = generated_dataset_path
        self.generated_radar_data_folder = generated_radar_data_folder
        self.generated_lidar_data_foler = generated_lidar_data_folder

        #check for generated dataset folder
        self._check_for_directory(generated_dataset_path, clear_contents=False)

        #check for radar and lidar directories
        self._check_for_directory(
            path= os.path.join(generated_dataset_path,generated_radar_data_folder),
            clear_contents=clear_existing_data
        )

        self._check_for_directory(
            path= os.path.join(generated_dataset_path,generated_lidar_data_folder),
            clear_contents=clear_existing_data
        )

        self.generated_file_name = generated_file_name

        #set the radar and lidar data paths
        self.radar_data_processor.init_save_file_paths(
            save_file_folder=os.path.join(self.generated_dataset_path,self.generated_radar_data_folder),
            save_file_name=self.generated_file_name
        )

        self.lidar_data_processor.init_save_file_paths(
            save_file_folder=os.path.join(self.generated_dataset_path,self.generated_lidar_data_foler),
            save_file_name=self.generated_file_name
        )
        
        self.generated_dataset_paths_set = True

        #set the number of currently generated samples
        self.num_samples = len(
            os.listdir(os.path.join(self.generated_dataset_path,self.generated_radar_data_folder))
        )

        return
    
    def _check_for_directory(self,path, clear_contents = False):
        """Checks to see if a directory exists, 
        if the directory does not exist, attepts to create the directory.
        If it does exist, optionally removes all files

        Args:
            path (str): path to the directory to create
            clear_contents (bool, optional): removes all contents in the directory on True. Defaults to False.
        """

        if os.path.isdir(path):
            print("DatasetGenerator._check_for_directory: found directory {}".format(path))

            if clear_contents:
                print("DatasetGenerator._check_for_directory: clearing contents of {}".format(path))

                #clear the contents
                for file in os.listdir(path):
                    file_path = os.path.join(path,file)

                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print("Failed to delete {}".format(path))
        else:
            print("DatasetGenerator._check_for_directory: creating directory {}".format(path))
            os.makedirs(path)
        return

    def config_radar_lidar_data_paths_from_csv(
            self,
            scenario_folder,
            radar_col_name,
            lidar_col_name
    ):
        """Configure the radar and lidar data paths from a .csv file located in a 
        scenario folder (For Deepsense 6G dataset)

        Args:
            scenario_folder (str): the path to the scenario folder that the .csv file is located inside
            radar_col_name (str): the name of the column in the .csv file corresponding to the paths for the radar data
            lidar_col_name (str): the name of the column in the .csv file corresponding to the paths for the lidar data

        Raises:
            Exception: when the .csv file is not found
        """

        try:
            csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
            csv_path = os.path.join(scenario_folder, csv_file)
        except:
            raise Exception(f'No csv file inside {scenario_folder}.')
        
        dataframe = pd.read_csv(csv_path)
        print(f'Columns: {dataframe.columns.values}')
        print(f'Number of Rows: {dataframe.shape[0]}')
        
        #set the radar data paths
        radar_rel_paths = dataframe[radar_col_name].values
        radar_rel_paths = np.char.replace(radar_rel_paths.astype(str),"./",'')

        self.radar_data_paths = [os.path.join(
            scenario_folder,relative_path) \
            for relative_path in radar_rel_paths
        ]

        #set the lidar data paths
        lidar_rel_paths = dataframe[lidar_col_name].values
        lidar_rel_paths = np.char.replace(lidar_rel_paths.astype(str),"./",'')

        self.lidar_data_paths = [os.path.join(
            scenario_folder,relative_path) \
            for relative_path in lidar_rel_paths
        ]

        self.lidar_radar_data_paths_set = True
        
        return
    
    def config_radar_lidar_data_paths(
            self,
            scenario_folder,
            radar_data_folder,
            lidar_data_folder
    ):
        """Configure the radar and lidar data paths (for CPSL datasets)

        Args:
            scenario_folders (str): path to scenario the folder containing data used to generate datasets
            radar_data_folder (str): the name of the folder with the radar data
            lidar_data_folder (str): the name of the folder with the lidar data
        """
            
        #get the radar relative paths
        self.radar_data_paths = [os.path.join(
            scenario_folder,
            radar_data_folder,file) for \
            file in sorted(os.listdir(
            os.path.join(scenario_folder,radar_data_folder)))]
        
        #get the lidar relative paths
        self.lidar_data_paths = [os.path.join(
            scenario_folder,
            lidar_data_folder,file) for \
            file in sorted(os.listdir(
            os.path.join(scenario_folder,lidar_data_folder)))]
        
        print("DatasetGenerator.config_radar_lidar_data_paths: found {} samples".format(len(self.radar_data_paths)))

        self.radar_data_processor.init_radar_data_paths(self.radar_data_paths)
        self.lidar_data_processor.init_lidar_data_paths(self.lidar_data_paths)

        self.lidar_radar_data_paths_set = True
        
        return

    def config_radar_data_processor(
            self,
            max_range_bin:int,
            num_chirps_to_save:int,
            radar_fov:list,
            num_angle_bins:int,
            power_range_dB:list,
            chirps_per_frame,
            rx_channels,
            tx_channels,
            samples_per_chirp,
            adc_sample_rate_Hz,
            chirp_slope_MHz_us,
            start_freq_Hz,
            idle_time_us,
            ramp_end_time_us,
            num_previous_frames
    ):

        #configure the radar data processor
        self.radar_data_processor.configure(
            max_range_bin,
            num_chirps_to_save,
            radar_fov,
            num_angle_bins,
            power_range_dB,
            chirps_per_frame,
            rx_channels,
            tx_channels,
            samples_per_chirp,
            adc_sample_rate_Hz,
            chirp_slope_MHz_us,
            start_freq_Hz,
            idle_time_us,
            ramp_end_time_us,
            num_previous_frames
        )

        return

    def config_lidar_data_processor(
            self,
            max_range_m:float = 100,
            num_range_bins:int = 256,
            angle_range_rad:list=[0,np.pi],
            num_angle_bins:int = 256,
            num_previous_frames:int=0
    ):

        #configure the lidar data processor
        self.lidar_data_processor.configure(
            max_range_m,
            num_range_bins,
            angle_range_rad,
            num_angle_bins,
            num_previous_frames
        )

        return
    
    def generate_dataset(self, clear_contents = True):
        """Generate a dataset from a single scenario folder

        Args:
            clear_contents (bool, optional): on True, clears existing saved datasets. Defaults to True.

        Raises:
            DataSetPathsNotLoaded: Raised if the datapaths to the raw sensor data and save locations haven't been set
        """

        #configure the dataset paths
        if self.lidar_radar_data_paths_set and self.generated_dataset_paths_set:
            warnings.filterwarnings("ignore")

            print("DatasetGenerator.generate_dataset: Generating radar dataset")
            self.radar_data_processor.generate_and_save_all_grids(clear_contents = clear_contents)

            print("DatasetGenerator.generate_dataset: Generating Lidar Dataset")
            self.lidar_data_processor.generate_and_save_all_grids(clear_contents = clear_contents)
        
            self.num_samples = len(
            os.listdir(os.path.join(self.generated_dataset_path,self.generated_radar_data_folder))
        )

        else:
            raise DataSetPathsNotLoaded
    
    def generate_dataset_from_multiple_scenarios(
            self,
            scenario_folders:list,
            radar_data_folder,
            lidar_data_folder
    ):
        """Generate a dataset from multiple different scenarios

        Args:
            scenario_folders (list): a list of paths to scenario folders containing data used to generate datasets
            radar_data_folder (str): the name of the folder with the radar data
            lidar_data_folder (str): the name of the folder with the lidar data
        """

        for i in range(len(scenario_folders)):
            
            if i == 0:
                clear_contents = True
            else:
                clear_contents = False
            
            print("\n\nDatasetGenerator.generate_dataset_from_multiple_scenarios: generating dataset from scenario {} of {}: {}".format(i + 1, len(scenario_folders), scenario_folders[i]))
            #set the radar and lidar data paths
            self.config_radar_lidar_data_paths(
                scenario_folder=scenario_folders[i],
                radar_data_folder=radar_data_folder,
                lidar_data_folder=lidar_data_folder
            )

            #generate the dataset
            self.generate_dataset(clear_contents= clear_contents)

#plotting
# use this function, it can show images, I am trying to alao save jpgs when call here

    def plot_radar_lidar_data(
            self,
            sample_idx,
            axs = [],

            show = True

    ):
        """Plot the radar and lidar data in cartesian and spherical coordinates

        Args:
            sample_idx (int): The sample index,
            axs (Axes): the axes upon which to plot the data.
                Note: axes must have at least 2 rows and 2 columns
            show (bool): on True shows the plot. Defaults to True
        """
        #create the axes
        if len(axs) == 0:
            fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
            fig.subplots_adjust(wspace=0.2,hspace=0.3)

        self.radar_data_processor.plot_range_azimuth_response(
            sample_idx=sample_idx,
            ax_cartesian=axs[0,0],
            ax_spherical=axs[1,0],
            show=False
            # show=True
        )

        self.lidar_data_processor.plot_pointcloud(
            sample_idx=sample_idx,
            ax_cartesian=axs[0,1],
            ax_spherical=axs[1,1],
            show=False
            # show=True
        )

        if show:
            # plt.savefig("xiao_test.jpg")
            # print("save images 3")
            # plt.show()
            jpg_file_path = os.path.join(self.generated_dataset_path,"{}_{}.jpg".
                                         format(self.generated_file_name, sample_idx))
            
            plt.savefig(jpg_file_path)
            print("save images at ", jpg_file_path)
            # plt.show()
            # plt.close()
        
    def plot_saved_radar_lidar_data(
            self,
            sample_idx:int,
            axs = [], 
            show = True):
        """Generate a plot in spherical and cartesian from a previously saved spherical grid

        Args:
            sample_idx (int): The sample index,
            axs (Axes): the axes upon which to plot the data.
                Note: axes must have at least 2 rows and 2 columns
            show (bool): on True shows the plot. Defaults to True
        """
        #create the axes
        if len(axs) == 0:
            fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
            fig.subplots_adjust(wspace=0.2,hspace=0.3)

        self.radar_data_processor.plot_from_saved_range_azimuth_response(
            sample_idx=sample_idx,
            ax_cartesian=axs[0,0],
            ax_spherical=axs[1,0],
            show= False
        )

        self.lidar_data_processor.plot_from_saved_grid(
            sample_idx=sample_idx,
            ax_cartesian=axs[0,1],
            ax_spherical=axs[1,1],
            show=False
        )

        # if show:
            # plt.show()


#loading datasets from a file
    def load_saved_radar_lidar_data(self,sample_idx):
        """Load a previously generated radar rng-az response and lidar point cloud

        Args:
            sample_idx (int): sample index to get the data from

        Returns:
            _type_: radar_rng_az_resp, lidar_spherical_pointcloud
        """

        radar_rng_az_resp = self.radar_data_processor.load_range_az_spherical_from_file(sample_idx)
        lidar_spherical_pointcloud = self.lidar_data_processor.load_grid_from_file(sample_idx)

        return radar_rng_az_resp,lidar_spherical_pointcloud

class DataSetPathsNotLoaded(Exception):

    def __init__(self):
        self.message = "Attempted to perform operations without first setting the dataset paths"
        super().__init__()