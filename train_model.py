from CPSL_Radar.models.unet import unet
from CPSL_Radar.trainer import Trainer

#loss functions
from CPSL_Radar.losses.BCE_dice_loss import BCE_DICE_Loss
from CPSL_Radar.losses.dice_loss import DiceLoss
from CPSL_Radar.losses.focal_loss import FocalLoss

#other torch functions
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
import sys

def main():
    #initialize the unet
    unet_model = unet(
        encoder_input_channels= 41,
        encoder_out_channels= (64,128,256,512),
        decoder_input_channels= (1024,512,256,128),
        decoder_out_channels= 64,
        output_channels= 1,
        retain_dimmension= False,
        input_dimmensions= (64,48)
    )

    #initialize the transforms to use
    unet_transforms = [
        transforms.ToTensor(),
        transforms.Resize((64,48))
    ]

    #initialize the model training
    model_trainer = Trainer(
        model= unet_model,
        dataset_path= "/data/david/CPSL_Ground/train",
        input_directory="radar",
        output_directory="lidar",
        test_split= 0.15,
        working_dir="working_dir",
        transforms_to_apply=unet_transforms,
        batch_size= 1024,
        epochs=40,
        learning_rate=0.001,
        loss_fn= BCEWithLogitsLoss(),
        cuda_device='cuda:1',
        multiple_GPUs=False
    )

    #train the model
    model_trainer.train_model()

#create the controller object
if __name__ == '__main__':

    main()
    sys.exit()