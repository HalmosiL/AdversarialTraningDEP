from models.Network import PSPNet, Dummy
import torch
import torch.nn as nn

'''
def slice_model(model, level="Encoder"):
    if(level == "Encoder"):
        return model.getSliceModel().eval()

def get_model(device):
    return Dummy().to(device)

def load_model(path, device):
    return Dummy().to(device)

def load_model_slice(path, device):
    model = Dummy().getSliceModel().eval()
    model = model.to(device)
    return model

'''
def slice_model(model, level="Encoder"):
    if(level == "Encoder"):
        return model.getSliceModel().eval()

def get_model(device):
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=19,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True
    )

    model = model.to(device)
    return model

def load_model(path, device):
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=19,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    return model

def load_model_slice(path, device):
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=19,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True
    ).getSliceModel().eval()

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    return model


