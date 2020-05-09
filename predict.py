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
      1. checkpoint as --checkpoint with default value "./checkpoint.pth"
      2. image_directory as --im_dir
      3. top_k as --topk with default value 5
      4. category_names as --catname with default value "cat_to_name.json"
      5. with_gpu as --gpu with default value True
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="./checkpoint.pth", help="loading the trained model")
    parser.add_argument("--im_dir", type=str, help="location of the data")
    parser.add_argument("--topk", type=int, default=5, help="number of classes predicted and their probabilities")
    parser.add_argument("--catname", type=str, default="cat_to_name.json")
    parser.add_argument("--gpu", type=str, default=True)
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()



def main():
    user_input = get_input_args()

    model = modfunc.load_checkpoint(checkpoint_file=user_input.checkpoint)
    cat_to_name = load_cat_name(user_input.catname)
    probs, top_class = predict(image_path=user_input.im_dir, model=model, topk=user_input.topk, with_gpu=user_input.gpu)

    for cl, prob in zip(top_class, probs):
        print("{} : {}".format(cat_to_name[cl], prob))



def load_cat_name(cat_to_name_file):
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name



def predict(image_path, model, topk=5, with_gpu=True):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    #load the image
    image = utilfunc.process_image(image_path)

    #unsqueeze to get a 1 dimensionnal tensor
    image.unsqueeze_(0)

    model.eval()

    if with_gpu == True:
        model.to("cuda")
        image = image.to("cuda")

    #use the network
    with torch.no_grad():
        output = torch.exp(model(image))

    #predict the top5 most likely labels
    top_p, top_class = output.topk(topk)

    #convert the indices to classes
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}

    probs = top_p.cpu().numpy()[0]
    classes = list()

    for tclass in top_class.cpu().numpy()[0]:
        classes.append(idx_to_class[tclass])

    return probs, classes



def view_predictions(image_path, model):
    #predictions
    probs, top_class = predict(image_path, model)
    labels = [cat_to_name[tclass] for tclass in top_class]

    img = process_image(image_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(4,8), nrows=2)
    #plot the flower
    with Image.open(image_path) as img:
        ax1.imshow(img)
    ax1.set_title(cat_to_name[image_path.split('/')[2]]);

    #plot the predictions
    y_pos = np.arange(len(top_class))
    ax2.barh(y_pos, probs, color=(0, 0, 1))
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_title("Class probability");


if __name__ == "__main__":
    main()
