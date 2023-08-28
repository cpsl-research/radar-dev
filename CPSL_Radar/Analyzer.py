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
from scipy.spatial.distance import cdist

#dataset generator
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator

class Analyzer:

    font_size_title = 14
    font_size_axis_labels = 12
    line_width = 3

    def __init__(self,
                 dataset_generator:DatasetGenerator,
                 transforms_to_apply:list = None,
                 working_dir = "working_dir",
                 model_file_name = "trained.pth",
                 cuda_device = "cuda:0"):

        if torch.cuda.is_available():

            self.device = cuda_device
            torch.cuda.set_device(self.device)
        else:
            self.device = "cpu"
        
        #set the dataset generator
        self.dataset_generator = dataset_generator
        
        self.model:Module = None
        self.transforms = None
        self.working_dir = working_dir
        self.model_file_name = model_file_name

        #initialize the model
        self._init_model(transforms_to_apply=transforms_to_apply)

        #temp directory folder name
        self.temp_directory_path = "temp_CPSL_Radar"
        self.temp_file_name = "frame"

        #tracking failed predictions (predictions where nothing was predicted)
        self.num_failed_predictions = 0
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

### Performing Quantative Results
    
    def plot_chamfer_hausdorf_cdfs(self):
        
        #compute chamfer and hausdorf distances
        chamfer_distances,hausdorf_distances = self.compute_all_chamfer_hausdorff_distances()

        #create the figure
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot()

        #add chamfer to plot
        self._plot_cdf(
            distances=chamfer_distances,
            label="Chamfer Distance",
            show=False,
            percentile=0.95,
            ax = ax
        )

        #add hausdorf distance
        self._plot_cdf(
            distances=hausdorf_distances,
            label="Hausdorf Distance",
            show=False,
            percentile=0.95,
            ax = ax
        )

        plt.legend()
        plt.show()
    
    def compute_all_chamfer_hausdorff_distances(self):

        #initialize arrays to store the distributions in
        chamfer_distances = np.zeros((self.dataset_generator.num_samples))
        hausdorff_distances = np.zeros((self.dataset_generator.num_samples))

        #reset failed sample tracking
        self.num_failed_predictions = 0

        #compute the distances for each of the arrays
        print("Analyzer.compute_all_chamfer_hausdorff_distances: Computing chamfer and hausdorff distances")
        for i in tqdm(range(self.dataset_generator.num_samples)):
            chamfer_distances[i],hausdorff_distances[i] = self.compute_chamfer_hausdorff_distances(sample_idx=i,print_result=False)
        
        print("Analyzer.compute_all_chamfer_hausdorff_distances: number failed predictoins {} of {} ({}%)".format(
            self.num_failed_predictions,
            self.dataset_generator.num_samples,
            float(self.num_failed_predictions) / float(self.dataset_generator.num_samples)
        ))
        return chamfer_distances,hausdorff_distances
    
    def compute_chamfer_hausdorff_distances(self,sample_idx, print_result = False):
        """Returns the chamfer and hausdorff distances between the points in the ground truth point cloud and predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compute distances for
            print_result (bool, optional): On True, prints the distances. Defaults to False.

        Returns:
            double,double: Chamfer distance (m), Hausdorff distance (m)
        """

        try: 
            distances = self._compute_euclidian_distances(sample_idx)

            chamfer = self._compute_chamfer(distances)
            hausdorff = self._compute_hausdorff(distances)

            if print_result:
                print("Chamfer: {}, Hausdorff: {}".format(chamfer,hausdorff))

            return chamfer,hausdorff
        except ValueError:
            self.num_failed_predictions += 1
            return 0,0
    
    def _compute_euclidian_distances(self, sample_idx):
        """Compute the euclidian distance between all of the points in the ground truth point cloud and the predicted point cloud

        Args:
            sample_idx (int): The sample index of the point cloud to compare

        Returns:
            ndarray: an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud
        """
        original_input = self.dataset_generator.radar_data_processor.load_range_az_spherical_from_file(sample_idx)

        #get the ground truth grid, convert to spherical points, convert to cartesian points
        ground_truth = self.dataset_generator.lidar_data_processor.load_grid_from_file(sample_idx)
        ground_truth = self.dataset_generator.lidar_data_processor.grid_to_spherical_points(ground_truth)
        ground_truth = self.dataset_generator.lidar_data_processor._convert_spherical_to_cartesian(ground_truth)

        #get the prediction, convert to spherical points, convert to cartesian points
        prediction = self._make_prediction(original_input)
        prediction = self.dataset_generator.lidar_data_processor.grid_to_spherical_points(prediction)
        prediction = self.dataset_generator.lidar_data_processor._convert_spherical_to_cartesian(prediction)

        return cdist(ground_truth,prediction,metric="euclidean")
    
    def _compute_hausdorff(self,distances):
        """Compute the Hausdorff distance between the predicted point cloud and the ground truth point cloud
            Note: formula from: https://pdal.io/en/latest/apps/hausdorff.html
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: hausdorff distance
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        return  np.max([np.max(ground_truth_mins),np.max(prediction_mins)])
    
    def _compute_chamfer(self,distances):
        """Compute the Chamfer distance between the predicted point cloud and the ground truth point cloud
            Note: formula from: https://github.com/DavidWatkins/chamfer_distance
        Args:
            distances (ndarray): an N x M ndarray with the euclidian distance between the N points in the ground truth point cloud and M points in the predicted point cloud

        Returns:
            double: Chamfer distance
        """

        ground_truth_mins = np.min(distances,axis=1)
        prediction_mins = np.min(distances,axis=0)

        #square the distances
        ground_truth_mins = np.square(ground_truth_mins)
        prediction_mins = np.square(prediction_mins)

        return np.mean(ground_truth_mins) + np.mean(prediction_mins)

    def _plot_cdf(
            self,
            distances:np.ndarray,
            label:str,
            show=True,
            percentile = 0.95,
            ax:plt.Axes = None):

        if not ax:
            fig = plt.figure(figsize=(3,3))
            ax = fig.add_subplot()

        sorted_data = np.sort(distances)
        p = 1. * np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        #compute the index of the percentile
        idx = (np.abs(p - percentile)).argmin()

        plt.plot(sorted_data[:idx],p[:idx],
                 label=label,
                 linewidth=Analyzer.line_width)

        ax.set_xlabel('Error (m)',fontsize=Analyzer.font_size_axis_labels)
        ax.set_ylabel('CDF',fontsize=Analyzer.font_size_axis_labels)
        ax.set_title("Error Comparison",fontsize=Analyzer.font_size_title)

        if show:
            plt.legend()
            plt.show()
    
### Visualizing Results
    def view_result(self,sample_idx, axs = [], show = True):

        if len(axs)==0:
            fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
            fig.subplots_adjust(wspace=0.2,hspace=0.4)
        
        #get the input/output data
        original_input = self.dataset_generator.radar_data_processor.load_range_az_spherical_from_file(sample_idx)
        # original_output = self.dataset_generator.lidar_data_processor.load_grid_from_file(sample_idx)

        #plot the data from the datset
        self.dataset_generator.plot_saved_radar_lidar_data(
            sample_idx=sample_idx,
            axs=axs,
            show=False
        )

        #fix the plot titles
        axs[0,1].set_title('Ground Truth\nLidar Point Cloud (Cartesian)',fontsize=Analyzer.font_size_title)
        axs[1,1].set_title('Ground Truth\nLidar Point CLoud (Spherical)',fontsize=Analyzer.font_size_title)
        
        #get the prediction
        prediction = self._make_prediction(original_input)
        
        #plot the comparison
        self._plot_prediction(
            pred_lidar=prediction,
            ax_cartesian=axs[0,2],
            ax_spherical=axs[1,2]
        )

        if show:
            plt.show()
    
    def save_video(self,video_file_name,fps = 10):

        #initialize the temporary buffer
        self._create_temp_dir()
        
        #generate all result frames
        self._save_all_result_frames_to_temp()

        #generate movie
        self._generate_movie_from_saved_frames(video_file_name,fps)

        #delete the temp directory
        self._delete_temp_dir()

    
    def _save_all_result_frames_to_temp(self):

        #initialize the figure
        fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
        fig.subplots_adjust(wspace=0.2,hspace=0.4)

        print("Analyzer._save_all_result_frames_to_temp: saving result frames to {}".format(self.temp_directory_path))

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
            file_name = "{}_{}.png".format(self.temp_file_name,i + 10000)
            path = os.path.join(self.temp_directory_path,file_name)
            fig.savefig(path,format='png',dpi=50)
    
    def _generate_movie_from_saved_frames(self, video_file_name, fps):

        #initialize writer
        writer = imageio.get_writer(video_file_name,fps=int(fps))

        print("Analyzer._generate_movie_from_temp_frames: generating movie")
        #save each frame
        for i in tqdm(range(self.dataset_generator.num_samples)):

            #get the figure name
            file_name = "{}_{}.png".format(self.temp_file_name,i + 10000)
            path = os.path.join(self.temp_directory_path,file_name)

            writer.append_data(imageio.imread(path))
        
        writer.close()

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
        ax_cartesian.set_title('Predicted\nLidar Point Cloud (Cartesian)',fontsize=Analyzer.font_size_title)

        #plot points in spherical
        self.dataset_generator.lidar_data_processor._plot_grid_spherial(
            grid_spherical=pred_lidar,
            ax = ax_spherical,
            show=False
        )
        ax_spherical.set_title('Predicted\nLidar Point Cloud (Spherical)',fontsize=Analyzer.font_size_title)
        
        return

#managing the temp directory
    def _create_temp_dir(self):

        path = self.temp_directory_path
        if os.path.isdir(path):

            print("Analyzer._create_temp_dir: found temp dir: {}".format(path))

            #clear the temp directory
            self._clear_temp_dir()
        
        else:
            print("Analyzer._create_temp_dir: creating directory: {}".format(path))
            os.makedirs(path)
        
        return
    
    def _clear_temp_dir(self):

        path = self.temp_directory_path

        if os.path.isdir(path):
            print("Analyzer._clear_temp_dir: clearing temp directory {}".format(path))
            for file in os.listdir(path):

                file_path = os.path.join(path,file)

                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print("Failed to delete {}".format(file_path))
        
        else:
            print("Analyzer._clear_temp_dir: temp directory {} not found".format(path))

    def _delete_temp_dir(self):

        path = self.temp_directory_path

        if os.path.isdir(path):

            print("viewer._delete_temp_dir: deleting temp dir: {}".format(path))

            #clear the directory first
            self._clear_temp_dir()

            #delete the directory
            os.rmdir(path)
        
        else:
            print("Analyzer._delete_temp_dir: temp directory {} not found".format(path))
