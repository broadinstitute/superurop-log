import torchvision
import torch


# load vanilla(non-pretrained) FCOS model from torchvision
model = torchvision.models.detection.fcos_resnet50_fpn()