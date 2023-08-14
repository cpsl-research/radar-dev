# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
import cv2
import os


class Viewer:

    font_size_title = 14
    font_size_axis_labels = 12

    def __init__(self,
                 model:Module,
                 dataset_path,
                 input_directory = "radar",
                 output_directory = "lidar",
                 working_dir = "working_dir"):

        self.model
        pass

    def _plot_comparison(self,original_radar,original_lidar,pred_lidar):

        fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(15,5))

        #plot original radar map in spherical
        axs[0].imshow(original_radar,
                     cmap='binary')
        axs[0].set_ylabel("Range",fontsize=Viewer.font_size_axis_labels)
        axs[0].set_xlabel("Azimuth",fontsize=Viewer.font_size_axis_labels)
        axs[0].set_title("Original Radar (Spherical)",fontsize=Viewer.font_size_title)

        #plot original lidar map in spherical
        axs[1].imshow(original_lidar,
                     cmap='binary')
        axs[1].set_ylabel("Range",fontsize=Viewer.font_size_axis_labels)
        axs[1].set_xlabel("Azimuth",fontsize=Viewer.font_size_axis_labels)
        axs[1].set_title("Original Lidar (Spherical)",fontsize=Viewer.font_size_title)

        #plot predicted lidar map in spherical
        axs[2].imshow(pred_lidar,
                     cmap='binary')
        axs[2].set_ylabel("Range",fontsize=Viewer.font_size_axis_labels)
        axs[2].set_xlabel("Azimuth",fontsize=Viewer.font_size_axis_labels)
        axs[2].set_title("Predicted Lidar (Spherical)",fontsize=Viewer.font_size_title)
    
    def make_prediction(self,im)

