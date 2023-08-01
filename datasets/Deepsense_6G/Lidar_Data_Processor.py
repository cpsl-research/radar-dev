#import required python modules

import os
import open3d as o3d
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LidarDataProcessor:

    def __init__(self):
        
        #relative paths of lidar pointcloud data for each sample
        self.scenario_data_path:str = None
        self.lidar_rel_paths:np.ndarray = None
    
    def load_data_paths(self,scenario_data_path:str, relative_paths:np.ndarray):

        #load the path to the dataset scenario
        self.scenario_data_path = scenario_data_path

        #load the relative path to the lidar samples, and remove './' from each address
        self.lidar_rel_paths = np.char.replace(relative_paths.astype(str),'./','')
        return
    
    def get_point_cloud_points(self,sample_idx):

        path = os.path.join(self.scenario_data_path,self.lidar_rel_paths[sample_idx])
        print(path)
        cloud = o3d.io.read_point_cloud(path)

        return cloud.points

    def plot_point_cloud_cartesian(self,sample_idx, max_range, angle_range):

        points = np.asarray(self.get_point_cloud_points(sample_idx))

        #test convert to spherical
        points_spherical = self._convert_cartesian_to_spherical(points)
        
        points_spherical = self._filter_ranges_and_azimuths(points_spherical,
                                                            max_range,
                                                            angle_range)
        
        #x = r * sin(elevation) * cos(aximuth)
        points = self._convert_spherical_to_cartesian(points_spherical)
        

        # Generate a scatter plot of the points using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(points[:, 0], points[:, 1],s=0.5)
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
    
    def _filter_ranges_and_azimuths(self,points_spherical:np.ndarray,max_range:float,azimuth_range:list):
        """Filter values in a point cloud (spherical coordinates) that are within a specified maximum range 
        and specified azimuth range

        Args:
            points_spherical (np.ndarray): Nx3 array of points in spherical coordinates
            max_range (float): the maximum range of points in the point cloud
            azimuth_range (list): a min and max azimuth angle (specified as [min,max])
        """

        mask = (points_spherical[:,0] < max_range) & (points_spherical[:,1] > azimuth_range[0]) & (points_spherical[:,1] < azimuth_range[1])

        #filter out points not in radar's elevation beamwidth
        mask = mask & (np.abs(points_spherical[:,2] - np.pi/2) < 0.26)

        return points_spherical[mask]



