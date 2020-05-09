import json
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image



def load_image(path):
    """
    Loading an image and preprocessing it in order to use the network on it
    """
    print("Loading an image from :", path)

    #Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    img = Image.open(path)
    img_tensor = transformations(img)
    img_tensor.unsqueeze_(0)

    print("Image loaded...")

    return im_tensor




def create_dataloaders(path):
    """
    Defining the dataloader and the transformations for the trainig, validation and testing set
    """
    print("Creating the dataloaders from :", path)
    #Define the folders containing the images
    directory = {
            "train" : path + "/train",
            "test" : path + "/test",
            "valid" : path + "/valid"
            }

    #Define the transforms for the training, validation, and testing sets
    data_transforms = {
            "train" : transforms.Compose([transforms.RandomRotation(50),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
            "test" : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
            "valid" : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
            }

    #Load the datasets with ImageFolder
    image_datasets = {
             x : datasets.ImageFolder(directory[x], transform=data_transforms[x]) for x in ["train", "test", "valid"]
             }

    #Using the image datasets and the trainforms, define the dataloaders
    batch_sizes = {
            "train" : 32,
            "test" : 16,
            "valid" : 16
            }

    dataloaders = {
            x : torch.utils.data.DataLoader(image_datasets[x], batch_sizes[x], shuffle=True) for x in ["train", "test", "valid"]
            }

    print("Dataloaders created...")

    return image_datasets, dataloaders


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    im = Image.open(image)
    im_tensor = transformations(im)

    return im_tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.axis("off")

    return ax
