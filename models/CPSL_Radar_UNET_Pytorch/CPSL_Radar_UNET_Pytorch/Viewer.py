# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.transforms import Compose
import cv2
import os


class Viewer:

    font_size_title = 14
    font_size_axis_labels = 12

    def __init__(self,
                 dataset_path,
                 transforms_to_apply:list = None,
                 input_directory = "radar",
                 output_directory = "lidar",
                 sample_file_name = "frame",
                 working_dir = "working_dir",
                 model_file_name = "trained.pth"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model:Module = None
        self.transforms = None
        self.working_dir = working_dir
        self.model_file_name = model_file_name

        #initialize file paths
        self.dataset_path = dataset_path
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.sample_file_name = sample_file_name

        #initialize the model
        self._init_model(transforms_to_apply=transforms_to_apply)

        pass

    
    def _init_model(self,transforms_to_apply:list = None):

        #put the model into eval mode
        model_path = os.path.join(self.working_dir,self.model_file_name)
        self.model = torch.load(model_path).to(self.device)

        #put the model into eval mode
        self.model.eval()

        #compose the list of transforms
        if transforms_to_apply:
            self.transforms = Compose(transforms_to_apply)
        else:
            self.transforms = None

    
    def view_result(self,sample_idx):

        #get the path to the sample of interest
        input_file_path = os.path.join(self.dataset_path,self.input_directory,"{}_{}.npy".format(self.sample_file_name,sample_idx))
        output_file_path = os.path.join(self.dataset_path,self.output_directory,"{}_{}.npy".format(self.sample_file_name,sample_idx))

        #get the input/output data
        original_input = np.load(input_file_path)
        original_output = np.load(output_file_path)

        #get the prediction
        prediction = self._make_prediction(original_input)

        #plot the comparison
        self._plot_comparison(
            original_radar=original_input,
            original_lidar=original_output,
            pred_lidar=prediction
        )


        
    
    def _plot_comparison(self,
                         original_radar:np.ndarray,
                         original_lidar:np.ndarray,
                         pred_lidar:np.ndarray):
        """Plot a comparison between the original radar/lidar image, and the predicted lidar image

        Args:
            original_radar (np.ndarray): num range_bins x num az bins x num chirps normalized numpy array
            original_lidar (np.ndarray): num range bins x num angle bins original lidar mask
            pred_lidar (np.ndarray): num range bins x num angle bins predicted lidar mask
        """

        fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(15,5))

        #plot original radar map in spherical(the first chirp)
        axs[0].imshow(np.flip(original_radar[:,:,0],axis=0),
                     cmap='gray')
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
    
    def _make_prediction(self,original_radar:np.ndarray):

        #convert to float 32
        x = original_radar.astype(np.float32)

        with torch.no_grad():
        
            #apply transforms (i.e: convert to tensor)
            if self.transforms:
                x = self.transforms(x)
            
            #since only one sample, need to unsqueeze
            x = torch.unsqueeze(x,0)

            #send x to device
            x = x.to(self.device)
            
            #get the prediction and apply sigmoid
            pred = self.model(x).squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()

            #filter out weak predictions
            pred = (pred > 0.5) * 1.0

        return pred


