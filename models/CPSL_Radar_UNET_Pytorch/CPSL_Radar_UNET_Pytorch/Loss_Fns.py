from torch.nn import Module
from torch import sigmoid
from torch import Tensor

#import existing loss functions
from torch.nn import BCEWithLogitsLoss


class DiceLoss(Module):

    def __init__(self, dice_smooth = 1.0):

        super().__init__()
        self.dice_smooth = dice_smooth

    def forward(self, inputs:Tensor, outputs:Tensor):

        #pass the inputs through a sigmoid
        inputs = sigmoid(inputs)

        #flatten the inputs and outputs
        inputs = inputs.view(-1)
        outputs = outputs.view(-1)

        intersection = (inputs * outputs).sum()
        dice = (2.0 * intersection + self.dice_smooth)/(inputs.sum() + outputs.sum() + self.dice_smooth)

        return 1 - dice

class BCE_DICE_Loss(Module):

    def __init__(self, dice_weight = 0.1, dice_smooth = 1.0):

        super().__init__()
        self.dice_weight = dice_weight
        
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(dice_smooth=dice_smooth)
    def forward(self, inputs:Tensor, outputs:Tensor):

        return self.bce_loss(inputs,outputs) + (self.dice_weight * self.dice_loss(inputs,outputs))

