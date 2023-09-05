from networks import tvmodels, allmodels
import importlib


def get_model(network, num_classes, pretrained, get_features):
    if network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), network)
        if network == 'googlenet':
            network = tvnet(pretrained=pretrained, aux_logits=False)
        else:
            network = tvnet(pretrained=pretrained, num_classes=num_classes)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        network = net(pretrained=False, num_classes=num_classes, get_features=get_features)
    return network
