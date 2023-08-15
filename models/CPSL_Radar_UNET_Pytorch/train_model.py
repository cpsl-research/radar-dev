from CPSL_Radar_UNET_Pytorch.Model import unet
from CPSL_Radar_UNET_Pytorch.Model_Trainer import ModelTrainer
from CPSL_Radar_UNET_Pytorch.Loss_Fns import BCE_DICE_Loss
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss
import sys

def main():
    #initialize the unet
    unet_model = unet(
        encoder_input_channels= 3,
        encoder_out_channels= (32,64,128,256,512),
        decoder_input_channels= (1024,512,256,128,64),
        decoder_out_channels= 32,
        output_channels= 1,
        retain_dimmension= False,
        input_dimmensions= (128,128)
    )

    #initialize the transforms to use
    unet_transforms = [
        transforms.ToTensor(),
        transforms.Resize((128,128))
    ]

    #initialize the model training
    model_trainer = ModelTrainer(
        model= unet_model,
        dataset_path= "/data/david/DeepSense6G/scenario36/generated_dataset/",
        input_directory="radar",
        output_directory="lidar",
        test_split= 0.15,
        working_dir="working_dir",
        transforms_to_apply=unet_transforms,
        batch_size= 256,
        epochs=30,
        learning_rate=0.001,
        loss_fn= BCE_DICE_Loss(dice_weight=0.1,dice_smooth=1.0)
    )

    #train the model
    model_trainer.train_model()

#create the controller object
if __name__ == '__main__':

    main()
    sys.exit()