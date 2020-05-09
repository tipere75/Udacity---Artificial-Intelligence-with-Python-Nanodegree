from collections import OrderedDict
from torch import nn, optim
from torchvision import models
import torch


def create_model(model_architecture, output_size, change_classifier=False, nodes_per_hlayer=0):
    """
    Creating a classifier based on an architecture to choose among vgg13 and densenet201 architectures 
    The inputs are the number of nodes of each hidden layer
    We add a dropout an a ReLU function to each layer but the last one
    """
    if model_architecture == "vgg13":
        model = models.__dict__[model_architecture](pretrained=True)
        in_features = model.classifier[0].in_features
    elif model_architecture == "densenet201":
        model = models.__dict__[model_architecture](pretrained=True)
        in_features = model.classifier.in_features

    #freeze parameters so we don't modify them later
    for param in model.parameters():
        param.requires_grad = False

    #to change the classifier if requested
    if change_classifier == True:
        #definition of the nodes of each layer
        nodes = [in_features]
        for node in nodes_per_hlayer:
            nodes.append(node)
        nodes.append(output_size)
        nlayers = len(nodes)

        hidden_layers = []

        #definition of the classifier with dropout and ReLU function
        for i in range(nlayers - 2):
            hidden_layers.append(("dropout"+str(i+1), nn.Dropout()))
            hidden_layers.append(("fc"+str(i+1), nn.Linear(nodes[i], nodes[i+1])))
            hidden_layers.append(("relu"+str(i+1), nn.ReLU()))

        hidden_layers.append(("dropout"+str(nlayers-1), nn.Dropout()))
        hidden_layers.append(("fc"+str(nlayers-1), nn.Linear(nodes[-2], nodes[-1])))
        hidden_layers.append(("output", nn.LogSoftmax(dim=1)))

        classifier = nn.Sequential(OrderedDict(hidden_layers))

        model.classifier = classifier

    return model



def save_checkpoint(filename, model, image_datasets, output_size, architecture, hidden_layers, learning_rate, epochs, optimizer):
    """
    Saving the architecture and the parameters of the model, as well as the mapping of classes to indices
    """
    print("Save the model to :", filename)

    #save the mapping of classes to indices
    model.class_to_idx = image_datasets["train"].class_to_idx

    #save the architecture and the parameters
    checkpoint = {"architecture": architecture,
              "input size": model.classifier[1].in_features,
              "output size": output_size,
              "hidden layers": hidden_layers,
              "learning rate": learning_rate,
              "epochs": epochs,
              "optimizer_state_dict": optimizer.state_dict(),
              "class_to_idx": model.class_to_idx,
              "state_dict": model.state_dict()}

    torch.save(checkpoint, filename)

    print("Saving completed...")



def load_checkpoint(checkpoint_file):
    """
    Load the architecture and the parameters of a saved model
    """
    print("Loading the model from :", checkpoint_file)

    checkpoint = torch.load(checkpoint_file)
    model = models.__dict__[checkpoint["architecture"]](pretrained=True)

    #freeze parameters so we don't modify them later
    for param in model.parameters():
        param.requires_grad = False

    #definition of the classifier with dropout and ReLU function
    nodes = [checkpoint["input size"]]
    for hid in checkpoint["hidden layers"]:
        nodes.append(hid)
    nodes.append(checkpoint["output size"])
    nlayers = len(nodes)

    hidden_layers = []

    for i in range(nlayers - 2):
        hidden_layers.append(("dropout"+str(i+1), nn.Dropout()))
        hidden_layers.append(("fc"+str(i+1), nn.Linear(nodes[i], nodes[i+1])))
        hidden_layers.append(("relu"+str(i+1), nn.ReLU()))

    hidden_layers.append(("dropout"+str(nlayers-1), nn.Dropout()))
    hidden_layers.append(("fc"+str(nlayers-1), nn.Linear(nodes[-2], nodes[-1])))
    hidden_layers.append(("output", nn.LogSoftmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(hidden_layers))

    model.classifier = classifier

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    print("Model loaded...")

    return model
