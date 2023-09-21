from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import numpy as np


# used for image segmentation tasks -xiao   the mask is 0 or 1 grid
class SegmentationDataset(Dataset):

    def __init__(self,input_paths:list,mask_paths:list,transforms:list = None):
        """initialize the segmentation dataset

        Args:
            input_paths (list): list of paths (strings) to each input file
            mask_paths (list): list of paths (strings) to each output file (mask)
            transforms (list, optional): A list of transforms to be applied to each item when __getitem__ is called. Will be fed into a compose() method Defaults to None.
        """
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

        self.num_samples = len(input_paths)

    def __len__(self):
        """Get the number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a item from the dataset at the specified index

        Args:
            idx (int): _description_

        Returns:
            (torch.tensor,torch.tensor): a tuple containing the (image,mask)
        """

        #get the path to the sample image
        image_path = self.input_paths[idx]
        mask_path = self.mask_paths[idx]

        #load the data from the disk
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        #apply transforms as required
        if self.transforms:
            transforms = Compose(self.transforms)
            image = transforms(image)
            mask = transforms(mask)
            #TODO: did this to ensure that random transforms applied to both image and mask

        return (image,mask)
