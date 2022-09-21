from torch import nn
import timm
import cv2

class PogChampModel(nn.Module):
    def __init__(
        self,
        model_name:str = 'resnet18',
        num_classes:int  = 4,
        **kwargs,
    ) -> None:

        super().__init__()

        self.model = timm.create_model(model_name, pretrained = True,
                                        num_classes = num_classes,**kwargs)
    
    def forward(self,x):
        
        return self.model(x)