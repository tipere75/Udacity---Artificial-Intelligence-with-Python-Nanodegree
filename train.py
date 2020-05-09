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

import model_functions as modfunc
import utility_functions as utilfunc

import argparse


def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 7 command line arguments. If
    the user fails to provide some or all of the 7 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. architecture as --arch with default value "vgg13"
      2. data_directory as --data_dir with default value "./flowers"
      3. save_directory as --save_dir with default value "./checkpoint.pth"
      4. nodes_per_hlayer as --npl with default value [1000]
      5. output_size as --outsize with default value 102
      6. epochs as --epochs with default value 10
      7. learning_rate as --lr with default value 0.001
      8. with_gpu as --gpu with default value True
      9. change_classifier as --change_class with default value True
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, default="vgg13", help="which model to use")
    parser.add_argument("--data_dir", type=str, default="./flowers", help="location of the data")
    parser.add_argument("--save_dir", type=str, default="./checkpoint.pth", help="where to save the model")
    parser.add_argument("--npl", type=int, default=[1000], help="number of node in each hidden layer")
    parser.add_argument("--outsize", type=int, default=102, help="number of possible outputs (different classes)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="value of the learnin rate")
    parser.add_argument("--gpu", type=str, default=True, help="train the network on the gpu or not")
    parser.add_argument("--change_class", type=str, default=True, help="whether or not we keep the given classifier")
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()



def main():
    user_input = get_input_args()

    image_datasets, dataloaders = utilfunc.create_dataloaders(user_input.data_dir)
    model = modfunc.create_model(user_input.arch, output_size=user_input.outsize, change_classifier=user_input.change_class,
                            nodes_per_hlayer=user_input.npl)
    
    optimizer=optim.Adam(model.classifier.parameters(), lr=user_input.lr)
    criterion=nn.NLLLoss()

    network_training(model=model, dataloaders=dataloaders, epochs=user_input.epochs, learning_rate=user_input.lr,
                    with_gpu=user_input.gpu)

    modfunc.save_checkpoint(filename=user_input.save_dir, model=model, image_datasets=image_datasets,
                        architecture=user_input.arch, output_size=user_input.outsize,hidden_layers=user_input.npl,
                        learning_rate=user_input.lr, optimizer=optimizer, epochs=user_input.epochs)




def network_training(model, dataloaders, epochs, learning_rate,  with_gpu=True):
    """
    Function to train the model based on the trainloader and validloader
    """
    start_time = time.time()

    trainloader = dataloaders["train"]
    validloader = dataloaders["valid"]

    if with_gpu == True :
        model.to("cuda")

    print_every = 5
    steps = 0
    running_loss = 0

    optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion=nn.NLLLoss()

    model.train()

    for e in range(epochs):
        running_loss = 0

         #training part
        for images, labels in trainloader:
            steps += 1

            #Send the input into the GPU if requested
            if with_gpu == True:
                images, labels = images.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()

            #Training pass
            output = model.forward(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            #loss
            running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0

            model.eval()

            with torch.no_grad():
                for images, labels in validloader:

                    if with_gpu == True:
                        images, labels = images.to("cuda"), labels.to("cuda")

                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")

            running_loss = 0
            model.train()

    duration = time.time() - start_time

    print("Training completed in : {}h {}mn {}s...".format(int((duration/3600)%24), int((duration/60)%60), int(duration%60)))



def network_validation(model, dataloaders, with_gpu=True):
    """
    Function to do validation on the test sets
    """
    model.eval()

    testloader = dataloaders["test"]

    if with_gpu == True:
        model.to("cuda")

    nb_correct_predictions, nb_predictions = 0, 0

    for images, labels in testloader:
        if with_gpu == True:
            images, labels = images.to("cuda"), labels.to("cuda")

        logps = model.forward(images)
        ps = torch.exp(logps)

        top_p, top_class = ps.topk(1, dim=1)
        equals = (top_class == labels.view(*top_class.shape))

        nb_predictions += labels.size(0)
        nb_correct_predictions += torch.sum(equals.type(torch.FloatTensor)).item()

    accuracy = nb_correct_predictions / nb_predictions

    print("Accuracy on the testloader : {}%...".format(round(accuracy*100,1)))



if __name__ == "__main__":
    main()
