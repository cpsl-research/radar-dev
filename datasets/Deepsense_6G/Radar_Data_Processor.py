import os
import numpy as np
from scipy.constants import c,pi
import matplotlib.pyplot as plt
import imageio
import io
from IPython.display import display, clear_output
from tqdm import tqdm

class RadarDataProcessor:
    
    #global params

    #angular FFT params
    num_angle_bins = 64

    #plotting parameters
    font_size_title = 14
    font_size_axis_labels = 12
    font_size_color_bar = 10

    def __init__(self):
        
        #given radar parameters
        self.chirps_per_frame = None
        self.rx_channels = None
        self.tx_channels = None
        self.samples_per_chirp = None
        self.adc_sample_rate_Hz = None
        self.chirp_slope_MHz_us = None
        self.start_freq_Hz = None
        self.idle_time_us = None
        self.ramp_end_time_us = None

        #computed radar parameters
        self.chirp_BW_Hz = None

        #computed radar performance specs
        self.range_res = None
        self.range_bins = None
        self.phase_shifts = None
        self.angle_bins = None
        self.thetas = None
        self.rhos = None
        self.x_s = None
        self.y_s = None

        #raw radar cube for a single frame (indexed by [rx channel, sample, chirp])
        #TODO: add support for DCA1000 Data Processing

        #relative paths of raw radar ADC data
        self.scenario_data_path:str = None
        self.radar_rel_paths:np.ndarray = None
        self.save_file_folder:str = None
        self.save_file_name:str = None
        
        #plotting
        self.fig = None
        self.axs = None

        self.max_range_bin = 0
        self.num_chirps_to_save = 0
        self.num_angle_bins = 0
        self.power_range_dB = None #specified as [min,max]

        return

    def configure(self,
                    scenario_data_path:str,
                    radar_rel_paths:np.ndarray,
                    save_file_folder:str,
                    save_file_name:str,
                    max_range_bin:int,
                    num_chirps_to_save:int,
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
                    ramp_end_time_us):
        
        #save the data paths
        self._init_data_paths(
            scenario_data_path,
            radar_rel_paths,
            save_file_folder,
            save_file_name
        )
        
        #load the radar parameters
        self.max_range_bin = max_range_bin
        self.num_chirps_to_save = num_chirps_to_save
        self.num_angle_bins = num_angle_bins
        self.power_range_dB = power_range_dB
        self.chirps_per_frame = chirps_per_frame
        self.rx_channels = rx_channels
        self.tx_channels = tx_channels
        self.samples_per_chirp = samples_per_chirp
        self.adc_sample_rate_Hz = adc_sample_rate_Hz
        self.chirp_slope_MHz_us = chirp_slope_MHz_us
        self.start_freq_Hz = start_freq_Hz
        self.idle_time_us = idle_time_us
        self.ramp_end_time_us = ramp_end_time_us
        

        #init computed params
        self._init_computed_params()

        #print the max range
        print("max range: {}m".format(self.max_range_bin * self.range_res))

        return

    def _init_data_paths(self,
                        scenario_data_path:str,
                        radar_rel_paths:np.ndarray,
                        save_file_folder:str,
                        save_file_name:str):
        
        self.scenario_data_path = scenario_data_path

        #load the relative path to the radar samples, and remove the './' from the path
        self.radar_rel_paths = np.char.replace(radar_rel_paths.astype(str),'./','')
        
        self.save_file_folder = save_file_folder
        self.save_file_name = save_file_name
        return

    def _init_computed_params(self):

        #chirp BW
        self.chirp_BW_Hz = self.chirp_slope_MHz_us * 1e12 * self.samples_per_chirp / self.adc_sample_rate_Hz

        #range resolution
        self.range_res = c / (2 * self.chirp_BW_Hz)
        self.range_bins = np.arange(0,self.samples_per_chirp) * self.range_res

        #angular parameters
        self.phase_shifts = np.arange(pi,-pi  - 2 * pi /(self.num_angle_bins - 1),-2 * pi / (self.num_angle_bins-1))
        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi
        self.angle_bins = np.arcsin(self.phase_shifts / pi)
        
        #mesh grid coordinates for plotting
        self.thetas,self.rhos = np.meshgrid(self.angle_bins,self.range_bins[:self.max_range_bin])
        self.x_s = np.multiply(self.rhos,np.sin(self.thetas))
        self.y_s = np.multiply(self.rhos,np.cos(self.thetas))

    def plot_range_azimuth_response(self,sample_idx:int):
        """Plot the range-azimuth response in cartesian and spherical coordinates

        Args:
            sample_idx (int): The sample index
        """
        #setup the axes
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        fig.subplots_adjust(wspace=0.2)

        #get the raw ADC data cube
        adc_data_cube = self._get_raw_ADC_data_cube(sample_idx)

        #compute the frame range-azimuth response
        range_azimuth_response = self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)

        #plot the response in cartesian for the first chirp
        self._plot_range_azimuth_heatmap_cartesian(range_azimuth_response[:,:,0],
                                                   ax=axs[0],
                                                   show=False)
        
        #plot the response in spherical coordinates
        self._plot_range_azimuth_heatmap_spherical(range_azimuth_response[:,:,0],
                                                   ax=axs[1],
                                                   show=False)
        
        plt.show()
        return
    
    def plot_from_saved_range_azimuth_response(self,sample_idx:int):
        """Plot the range-azimuth response in cartesian and spherical coordinates
        from a previously saved response

        Args:
            sample_idx (int): The sample index
        """
        #setup the axes
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        fig.subplots_adjust(wspace=0.2)

        range_azimuth_response = self.load_range_az_spherical_from_file(sample_idx=sample_idx)

        #plot the response in cartesian for the first chirp
        self._plot_range_azimuth_heatmap_cartesian(range_azimuth_response[:,:,0],
                                                   ax=axs[0],
                                                   show=False)
        
        #plot the response in spherical coordinates
        self._plot_range_azimuth_heatmap_spherical(range_azimuth_response[:,:,0],
                                                   ax=axs[1],
                                                   show=False)
    
    def generate_and_save_range_azimuth_response(self,sample_idx:int):
        """Compute the range_azimuth response and save it to a file

        Args:
            sample_idx (int): The sample index for which to generate and save the result to
        """
        #get the raw ADC data cube
        adc_data_cube = self._get_raw_ADC_data_cube(sample_idx)

        #compute the frame range-azimuth response
        range_azimuth_response = self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)

        self._save_range_az_spherical_to_file(range_azimuth_response,sample_idx=sample_idx)

        return
    
    def generate_and_save_all_grids(self):
        """Save all of the loaded radar range-azimuth heatmaps to files
        """

        num_files = len(self.radar_rel_paths)

        for i in tqdm(range(num_files)):
            self.generate_and_save_range_azimuth_response(sample_idx=i)




    def load_data_from_DCA1000(self,file_path):
        
        #TODO: Need to update this function to support loading data in from the DCA1000
        #import the raw data
        LVDS_lanes = 4
        adc_data = np.fromfile(file_path,dtype=np.int16)

        #reshape to get the real and imaginary parts
        adc_data = np.reshape(adc_data, (LVDS_lanes * 2,-1),order= "F")

        #convert into a complex format
        adc_data = adc_data[0:4,:] + 1j * adc_data[4:,:]

        #reshape to index as [rx channel, sample, chirp, frame]
        adc_data_cube = np.reshape(adc_data,(self.rx_channels,self.samples_per_chirp,self.chirps_per_frame,-1),order="F")

    def _get_raw_ADC_data_cube(self,sample_idx:int):
        """Get the raw ADC data cube associated with the given data sample

        Args:
            sample_idx (int): the sample index to get the adc data cube for

        Returns:
            np.ndarray: the adc data cube indexed by (indexed by [rx channel, sample, chirp])
        """

        path = os.path.join(self.scenario_data_path,self.radar_rel_paths[sample_idx])

        return np.load(path)
    
    def _compute_frame_normalized_range_azimuth_heatmaps(self,adc_data_cube:np.ndarray):

        frame_range_az_heatmaps = np.zeros((self.max_range_bin,self.num_angle_bins,self.num_chirps_to_save))

        for i in range(self.num_chirps_to_save):
            frame_range_az_heatmaps[:,:,i] = self._compute_chirp_normalized_range_azimuth_heatmap(adc_data_cube,chirp=i)
    
        return frame_range_az_heatmaps
    
    def _compute_chirp_normalized_range_azimuth_heatmap(self,adc_data_cube:np.ndarray,chirp=0):
        """Compute the range azimuth heatmap for a single chirp in the raw ADC data frame

        Args:
            adc_data_cube (np.ndarray): _description_
            chirp (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: the computed range-azimuth heatmap (normalized and thresholded)
        """

        #get range angle cube
        data = np.zeros((self.samples_per_chirp,self.num_angle_bins),dtype=complex)
        data[:,0:self.rx_channels] = np.transpose(adc_data_cube[:,:,chirp])

        #compute Range FFT
        data = np.fft.fftshift(np.fft.fft(data,axis=0))

        #compute range response
        data = 20* np.log10(np.abs(np.fft.fftshift(np.fft.fft(data,axis=-1))))

        #[for debugging] to get an idea of what the max should be
        max_db = np.max(data)
        
        #filter to only output the desired ranges
        data = data[:self.max_range_bin,:]

        #perform thresholding on the input data
        data[data <= self.power_range_dB[0]] = self.power_range_dB[0]
        data[data >= self.power_range_dB[1]] = self.power_range_dB[1]
        
        #normalize the data
        data = (data - self.power_range_dB[0]) / \
            (self.power_range_dB[1] - self.power_range_dB[0])

        return data
    
    def _plot_range_azimuth_heatmap_cartesian(self,
                                              rng_az_response:np.ndarray,
                                              ax:plt.Axes=None,
                                              show=True):
        """Plot the range azimuth heatmap (for a single chirp) in cartesian coordinates

        Args:
            rng_az_response (np.ndarray): num_range_bins x num_angle_bins normalized range azimuth response
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
        
        cartesian_plot = ax.pcolormesh(
            self.x_s,
            self.y_s,
            rng_az_response[:self.max_range_bin,:],
            shading='gouraud',
            cmap="gray")
        ax.set_xlabel('X (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_ylabel('Y (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_title('Range-Azimuth\nHeatmap (Cartesian)',fontsize=RadarDataProcessor.font_size_title)

        if show:
            plt.show()
        #if enable_color_bar:
        #    cbar = self.fig.colorbar(cartesian_plot)
        #    cbar.set_label("Relative Power (dB)",size=RadarDataProcessor.font_size_color_bar)
        #    cbar.ax.tick_params(labelsize=RadarDataProcessor.font_size_color_bar)
    
    def _plot_range_azimuth_heatmap_spherical(self,
                                              rng_az_response:np.ndarray,
                                              ax:plt.Axes = None,
                                              show = True):
        """Plot the range azimuth heatmap in spherical coordinates

        Args:
            rng_az_response (np.ndarray): num_range_bins x num_angle_bins normalized range azimuth response
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """

        if not ax:
            fig = plt.fig()
            ax = fig.add_subplot()

        #plot polar coordinates
        max_range = self.max_range_bin * self.range_res
        ax.imshow(np.flip(rng_az_response),
                  cmap="gray",
                  extent=[self.angle_bins[-1],self.angle_bins[0],
                          self.range_bins[0],max_range],
                          aspect='auto')
        ax.set_xlabel('Angle(radians)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_ylabel('Range (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_title('Range-Azimuth\nHeatmap (Polar)',fontsize=RadarDataProcessor.font_size_title)

        #if enable_color_bar:
        #    cbar = self.fig.colorbar(polar_plt)
        #    cbar.set_label("Relative Power (dB)",size=RadarDataProcessor.font_size_color_bar)
        #    cbar.ax.tick_params(labelsize=RadarDataProcessor.font_size_color_bar)
        if show:
            plt.show()

#save to a file

    def _save_range_az_spherical_to_file(self,range_azimuth_response:np.ndarray,sample_idx:int):
        """Save the given range-azimuth response (in spherical) to a file at the configured location

        Args:
            range_azimuth_response (np.ndarray): The range azimuth response to save
            sample_idx (int): The index of the sample to be saved
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(self.save_file_name,sample_idx)
        path = os.path.join(self.save_file_folder,file_name)

        #save the file to a .npy array
        np.save(path,range_azimuth_response)

        return
    
    def load_range_az_spherical_from_file(self,sample_idx:int):
        """Load a previously saved range-azimuth response from a file

        Args:
            sample_idx (int): The sample index of the file to load

        Returns:
            np.ndarray: The loaded range-azimuth response
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(self.save_file_name,sample_idx)
        path = os.path.join(self.save_file_folder,file_name)

        #load the grid
        return np.load(path)


