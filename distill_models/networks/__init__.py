from torchvision import models
from .resnet32 import resnet32
from .CIFAR_resnet import *
from .mobilenet_v2 import mobilenet_v2
from .shufflenet import *
from .squeezenet import *

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v3_large',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

allmodels = tvmodels + ['resnet32', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'mobilenet_v2',
                        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                        'squeezenet1_0', 'squeezenet1_1']
