# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.transforms import Compose
import cv2
import os
from tqdm import tqdm
import io
import imageio

#dataset generator
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator

class Viewer:

    font_size_title = 14
    font_size_axis_labels = 12

    def __init__(self,
                 dataset_generator:DatasetGenerator,
                 transforms_to_apply:list = None,
                 working_dir = "working_dir",
                 model_file_name = "trained.pth"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #set the dataset generator
        self.dataset_generator = dataset_generator
        
        self.model:Module = None
        self.transforms = None
        self.working_dir = working_dir
        self.model_file_name = model_file_name

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

    
    def view_result(self,sample_idx, axs = [], show = True):

        if axs.size==0:
            fig,axs = plt.subplots(nrows=3,ncols=2,figsize=(10,15))
            fig.subplots_adjust(wspace=0.2,hspace=0.4)
        
        #get the input/output data
        original_input = self.dataset_generator.radar_data_processor.load_range_az_spherical_from_file(sample_idx)
        original_output = self.dataset_generator.lidar_data_processor.load_grid_from_file(sample_idx)

        #plot the data from the datset
        self.dataset_generator.plot_saved_radar_lidar_data(
            sample_idx=sample_idx,
            axs=axs,
            show=False
        )

        #fix the plot titles
        axs[1,0].set_title('Ground Truth\nLidar Point Cloud (Cartesian)',fontsize=Viewer.font_size_title)
        axs[1,1].set_title('Ground Truth\nLidar Point CLoud (Spherical)',fontsize=Viewer.font_size_title)
        
        #get the prediction
        prediction = self._make_prediction(original_input)
        
        #plot the comparison
        self._plot_prediction(
            pred_lidar=prediction,
            ax_cartesian=axs[2,0],
            ax_spherical=axs[2,1]
        )

        if show:
            plt.show()
    
    def save_video(self,video_file_name,frame_duration_s = 10e-3):

        image_frames = []

        #initialize the figure
        fig,axs = plt.subplots(nrows=3,ncols=2,figsize=(10,15))
        fig.subplots_adjust(wspace=0.2,hspace=0.4)

        #save each frame
        for i in tqdm(range(self.dataset_generator.num_samples)):
            
            #clear the axes
            for ax in axs.flat:
                ax.cla()

            #plot the result
            self.view_result(
                sample_idx=i,
                axs=axs,
                show=False
            )

            #save the figure
            buf = io.BytesIO()
            fig.savefig(buf,format='png',dpi=300)
            buf.seek(0)
            image_frames.append(imageio.imread(buf))

        #save the end result to a file
        imageio.mimsave(video_file_name,image_frames,duration=frame_duration_s)


    
    def _plot_prediction(self, pred_lidar:np.ndarray,ax_cartesian,ax_spherical):
        """Plot a comparison between the original radar/lidar image, and the predicted lidar image

        Args:
            pred_lidar (np.ndarray): num range bins x num angle bins predicted lidar mask
        """

        #convert the spherical grid to cartesian points
        points_spherical = self.dataset_generator.lidar_data_processor.grid_to_spherical_points(pred_lidar)
        points_cartesian = self.dataset_generator.lidar_data_processor._convert_spherical_to_cartesian(points_spherical)

        #plot points in cartesian
        self.dataset_generator.lidar_data_processor._plot_points_cartesian(
            points_cartesian=points_cartesian,
            ax=ax_cartesian,
            show=False
        )
        ax_cartesian.set_title('Ground Truth\nLidar Point Cloud (Cartesian)',fontsize=Viewer.font_size_title)

        #plot points in spherical
        self.dataset_generator.lidar_data_processor._plot_grid_spherial(
            grid_spherical=pred_lidar,
            ax = ax_spherical,
            show=False
        )
        ax_spherical.set_title('Ground Truth\nLidar Point Cloud (Spherical)',fontsize=Viewer.font_size_title)
        
        return
    
    def _make_prediction(self,original_radar:np.ndarray):

        #convert to float 32
        x = original_radar.astype(np.float32)

        with torch.no_grad():
        
            #apply transforms (i.e: convert to tensor)
            self.model.eval()
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


