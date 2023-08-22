#import required python modules

import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

class LidarDataProcessor:

    #global plotting parameters

    font_size_title = 14
    font_size_axis_labels = 12

    def __init__(self):
        
        #configuration status
        self.configured = False
        
        #relative paths of lidar pointcloud data for each sample
        self.lidar_data_paths:list = None
        self.save_file_folder:str = None
        self.save_file_name:str = None

        #ranges and range bins
        self.max_range_m = 0
        self.num_range_bins = 0
        self.range_res_m = 0
        self.range_bins = None

        #az angles and angle bins
        self.az_angle_range_rad = None
        self.num_angle_bins = 0
        self.az_angle_res_rad = 0
        self.az_angle_res_deg = 0
        self.az_angle_bins = None
    
    def configure(self,
                  max_range_m:float = 100,
                  num_range_bins:int = 256,
                  angle_range_rad:list=[0,np.pi],
                  num_angle_bins:int = 256):
        
        self.configured = True

        #configure azimuth angle bins
        self._init_az_angle_bins(angle_range_rad,num_angle_bins)

        #configure the range bins
        self._init_range_bins(max_range_m,num_range_bins)
        pass
    
    def init_data_paths(self,
                         lidar_data_paths:list,
                           save_file_folder:str,
                           save_file_name:str):
        """Initialize the paths to the lidar data

        Args:
            lidar_data_paths (list): the list of paths to each lidar data frame
            save_file_folder (str): path to the folder to save the computed grid
            save_file_name (str): the base name of the file to save (will be save_file_name_#.npy)
        """

        self.lidar_data_paths = lidar_data_paths
        self.save_file_folder = save_file_folder
        self.save_file_name = save_file_name
        return
    
    def _init_az_angle_bins(self,angle_range_rad:list,num_angle_bins:int):
        """Initialize the azimuth angular bins

        Args:
            angle_range_rad (list): The [min,max] azimuth angle for the point cloud to filter to
            num_angle_bins (int): The number of azimuth bins (affects lidar resolution)
        """
        #configure angular bins
        self.num_angle_bins = num_angle_bins

        self.az_angle_range_rad = angle_range_rad
        self.az_angle_res_rad = (self.az_angle_range_rad[1] - self.az_angle_range_rad[0]) / self.num_angle_bins
        self.az_angle_res_deg = self.az_angle_res_rad * 180 / np.pi

        self.az_angle_bins = np.flip(np.arange(self.az_angle_range_rad[0],self.az_angle_range_rad[1],self.az_angle_res_rad))

    
    def _init_range_bins(self,max_range_m:float,num_range_bins:int):
        """Initialize the range bins

        Args:
            max_range_m (float): The maximum range for the point cloud to filter points to
            num_range_bins (int): The number of range bins
        """
        
        #configure range bins
        self.num_range_bins = num_range_bins
        self.max_range_m = max_range_m
        self.range_res_m = self.max_range_m/self.num_range_bins

        self.range_bins = np.flip(np.arange(0,self.max_range_m,self.range_res_m))

    def plot_pointcloud(
            self,
            sample_idx:int, 
            ax_cartesian = None,
            ax_spherical = None,
            show = True):
        """Plot the pointcloud in cartesian and spherical coordinates

        Args:
            sample_idx (int): The sample index,
            ax_cartesian (Axes): axes to plot the cartesian plot on. Defaults to None
            ax_spherical (Axes): axes to plot the spherical plot on. Defaults to None
            show (bool): on True shows the plot. Defaults to True
        """

        #setup the axes
        if (ax_cartesian == None) or (ax_spherical == None):

            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            fig.subplots_adjust(wspace=0.2)

            ax_cartesian = axs[0]
            ax_spherical = axs[1]

        #get the cartesian point cloud
        points_cartesian = self._get_point_cloud_points(sample_idx)

        #convert to spherical, filter, convert back to cartesian
        points_spherical = self._convert_cartesian_to_spherical(points_cartesian)
        points_spherical = self._filter_ranges_and_azimuths(points_spherical)
        points_cartesian = self._convert_spherical_to_cartesian(points_spherical)

        #plot the points in cartesian
        self._plot_points_cartesian(points_cartesian,ax=ax_cartesian,show=False)

        #generate the spherical points as a grid
        grid = self.points_spherical_to_grid(points_spherical)
        self._plot_grid_spherial(grid,ax=ax_spherical,show=False)

        if show:
            plt.show()
        return
    
    def plot_from_saved_grid(
            self,
            sample_idx:int, 
            ax_cartesian = None,
            ax_spherical = None,
            show = True):
        """Generate a plot in spherical and cartesian from a previously saved spherical grid

        Args:
            sample_idx (int): The sample index,
            ax_cartesian (Axes): axes to plot the cartesian plot on. Defaults to None
            ax_spherical (Axes): axes to plot the spherical plot on. Defaults to None
            show (bool): on True shows the plot. Defaults to True
        """
        
        #setup the axes

        if (ax_cartesian == None) or (ax_spherical == None):

            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            fig.subplots_adjust(wspace=0.2)

            ax_cartesian = axs[0]
            ax_spherical = axs[1]

        #load the grid
        grid_spherical = self.load_grid_from_file(sample_idx)

        #convert to spherical and then to cartesian
        points_spherical = self.grid_to_spherical_points(grid_spherical)
        points_cartesian = self._convert_spherical_to_cartesian(points_spherical)

        #plot points in cartesian
        self._plot_points_cartesian(points_cartesian,ax=ax_cartesian,show=False)

        #plot the points in spherical from the grid
        self._plot_grid_spherial(grid_spherical,ax=ax_spherical,show=False)

        if show:
            plt.show()

    
    def generate_and_save_grid(self,sample_idx:int):
        """Computes and saves the spherical grid for the given sample index

        Args:
            sample_idx (int): the sample index for which to generate and save the grid
        """

        #get the cartesian point cloud
        points_cartesian = self._get_point_cloud_points(sample_idx)

        #convert to spherical, filter, convert back to cartesian
        points_spherical = self._convert_cartesian_to_spherical(points_cartesian)
        points_spherical = self._filter_ranges_and_azimuths(points_spherical)

        #generate the spherical points as a grid
        grid = self.points_spherical_to_grid(points_spherical)

        #save grid to a file
        self._save_grid_to_file(grid_spherical=grid,sample_idx=sample_idx)

    def generate_and_save_all_grids(self):
        """Save all of the loaded lidar point clouds to files
        """

        num_files = len(self.lidar_data_paths)

        for i in tqdm(range(num_files)):
            self.generate_and_save_grid(sample_idx=i)
        
        return
## Helper Functions

#getting a point cloud
    def _get_point_cloud_points(self,sample_idx):

        path = self.lidar_data_paths[sample_idx]

        if ".ply" in path:
            cloud = o3d.io.read_point_cloud(path)
        elif ".csv" in path:
            # read the array from the pandas array
            col = [' RANGE (mm)', ' SIGNAL', ' REFLECTIVITY', ' NEAR_IR'][0]
            df = pd.read_csv(path)
            xyz = np.stack((df[' X (mm)'].values, df[' Y (mm)'].values, df[' Z (mm)'].values)).T * 1e-3

            #initialize a new point cloud array
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(xyz)
            arr = np.array(df[col].values)

            #assign colors (from Deepsense code)
            # colors  Type: | float64 array of shape (num_points, 3), range [0, 1] , use numpy.asarray() to access
            percentile = 95
            norm_factor = np.percentile(np.array(df[col].values), percentile)
            colormap = matplotlib.colormaps['jet']        
            cloud.colors = o3d.utility.Vector3dVector(colormap(arr/norm_factor)[:,:3]) #:3 to remove the alpha
        
        elif ".npy" in path:
            #read the array from the .npy file
            xyz = np.load(path)[:,:3]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(xyz)
        
        points = np.asarray(cloud.points)
        ground_plane = np.min(points[:,2])
        non_ground_points = points[:,2] > -0.2
        points = points[non_ground_points,:]
        #filter out points not in radar's elevation beamwidth

        return points

# generate the point cloud
    def _plot_points_cartesian(self,points_cartesian:np.ndarray,ax:plt.Axes = None,show=True):
        """Plot a given point cloud with points specified in cartesian coordinates

        Args:
            points_cartesian (np.ndarray): The points to plot stored in a Nx3 array with [x,y,z]
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()

        #in the lidar data, x is pointing forward, while y is pointing left, thus we need to rotate the plot
        #rotation matrix
        rotation_matrix_deepSense = np.array([[0,1,0],
                                            [-1,0,0],
                                            [0,0,1]])
        
        rotation_matrix_CPSL_gnd = np.array([[-1,0,0],
                                             [0,-1,0],
                                             [0,0,1]])
        
        #rotate the points
        points_cartesian = np.dot(points_cartesian,rotation_matrix_CPSL_gnd)


        ax.scatter(points_cartesian[:, 0], points_cartesian[:, 1],s=0.5)
        ax.set_xlabel('Y (m)',fontsize=LidarDataProcessor.font_size_axis_labels)
        ax.set_ylabel('X (m)',fontsize=LidarDataProcessor.font_size_axis_labels)
        ax.set_title('Lidar Point Cloud (Cartesian)',fontsize=LidarDataProcessor.font_size_title)
        ax.set_ylim(0,)
        if show:
            plt.show()

    def _plot_grid_spherial(self,grid_spherical:np.ndarray,ax:plt.Axes=None,show=True):
        """Plot a given point cloud stored as a grid where the points are in spherical coordinates

        Args:
            grid_spherical (np.ndarray): A grid where the x-axis is angle(radians), y-axis is range(m).
                A value of 1 indicates a point is present, A value of 0 indicates no point is present
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax.imshow(grid_spherical,
                  cmap='binary',
                  extent=[self.az_angle_bins[0],self.az_angle_bins[-1],
                          self.range_bins[-1],self.range_bins[0]],
                aspect='auto')
        ax.set_ylabel('Range (m)',fontsize=LidarDataProcessor.font_size_axis_labels)
        ax.set_xlabel('Azimuth (rad)',fontsize=LidarDataProcessor.font_size_axis_labels)
        ax.set_title('Lidar Point Cloud (Spherical)',fontsize=LidarDataProcessor.font_size_title)
        
        if show:
            plt.show()

    def _convert_cartesian_to_spherical(self,points_cart:np.ndarray):
        """Convert an array of points stored as (x,y,z) to (range,azimuth, elevation).
        Note that azimuth = 0 degrees for points on the positive x-axis

        Args:
            points_cart (np.ndarray): Nx3 matrix of points in cartesian (x,y,z)

        Returns:
            (np.ndarray): Nx3 matrix of points in spherical (range, azimuth, elevation) in radians
        """
        ranges = np.sqrt(points_cart[:, 0]**2 + points_cart[:, 1]**2 + points_cart[:, 2]**2)
        azimuths = np.arctan2(points_cart[:, 1], points_cart[:, 0])
        elevations = np.arccos(points_cart[:, 2] / ranges)

        return  np.column_stack((ranges,azimuths,elevations))
        
    def _convert_spherical_to_cartesian(self,points_spherical:np.ndarray):
        """Convert an array of points stored as (range, azimuth, elevation) to (x,y,z)

        Args:
            points_spherical (np.ndarray): Nx3 matrix of points in spherical (range,azimuth, elevation)

        Returns:
            (np.ndarray): Nx3 matrix of  points in cartesian (x,y,z)
        """

        x = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.cos(points_spherical[:,1])
        y = points_spherical[:,0] * np.sin(points_spherical[:,2]) * np.sin(points_spherical[:,1])
        z = points_spherical[:,0] * np.cos(points_spherical[:,2])


        return np.column_stack((x,y,z))
    
    def _filter_ranges_and_azimuths(self,points_spherical:np.ndarray):
        """Filter values in a point cloud (spherical coordinates) that are within the configured maximum range 
        and specified azimuth range

        Args:
            points_spherical (np.ndarray): Nx3 array of points in spherical coordinates
        """

        mask = (points_spherical[:,0] < self.max_range_m) & \
                (points_spherical[:,1] > self.az_angle_range_rad[0]) &\
                (points_spherical[:,1] < self.az_angle_range_rad[1])

        #filter out points not in radar's elevation beamwidth
        mask = mask & (np.abs(points_spherical[:,2] - np.pi/2) < 0.26) #was 0.26

        return points_spherical[mask]

#generate grids/images from the point cloud
    def points_spherical_to_grid(self,points_spherical:np.ndarray):
        """Convert a point cloud in spherical coordinates to a grid for plotting/image generation

        Args:
            points_spherical (np.ndarray): Nx3 array of points in [range,az,el] coordinates. Assumes that points have already been filtered

        Returns:
            np.ndarray: num_range_bins x num_angle bins matrix with index = 1 if a point was located there
        """
        #define the out grid
        out_grid = np.zeros((self.num_range_bins,self.num_angle_bins))

        #identify the nearest point from the pointcloud
        r_idx = np.argmin(np.abs(self.range_bins - points_spherical[:,0][:,None]),axis=1)
        az_idx = np.argmin(np.abs(self.az_angle_bins - points_spherical[:,1][:,None]),axis=1)

        out_grid[r_idx,az_idx] = 1

        return out_grid
    
    def grid_to_spherical_points(self,grid:np.ndarray):
        
        #get the nonzero coordinates
        rng_idx,az_idx = np.nonzero(grid)

        rng_vals = self.range_bins[rng_idx]
        az_vals = self.az_angle_bins[az_idx]
        el_vals = np.ones_like(rng_vals) * np.pi/2 #hard coded for now

        return np.column_stack((rng_vals,az_vals,el_vals))

#saving to a file
    def _save_grid_to_file(self,grid_spherical:np.ndarray,sample_idx:int):
        """Save the given spherical grid to a file at the configured location

        Args:
            grid_spherical (np.ndarray): The grid to save
            sample_idx (int): The index of the sample to be saved
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(self.save_file_name,sample_idx + 10000)
        path = os.path.join(self.save_file_folder,file_name)

        #save the file to a .npy array
        np.save(path,grid_spherical)

    def load_grid_from_file(self,sample_idx:int):
        """Load a previously saved grid from a file

        Args:
            sample_idx (int): The sample index of the file to load

        Returns:
            np.ndarray: The loaded grid
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(self.save_file_name,sample_idx + 10000)
        path = os.path.join(self.save_file_folder,file_name)

        #load the grid
        return np.load(path)





