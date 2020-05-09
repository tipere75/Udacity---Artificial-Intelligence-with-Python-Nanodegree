# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I first developped code for an image classifier built with PyTorch, then converted it into a command line application.


## Projects assets :
- Image Classifier Project.ipynb : Jupyter notebook containing my code for creating, training and saving a neural network, as well as using it to predict the name of a flower
- Image Classifier Project.html : html export of the Jupyter notebook
- train.py : command line application to train a network on a dataset
- predict.py : command line application to predict flower name from an image
- model_functions.py : functions to create, save or load a model
- utility_functions.py : functions to load data and preprocess images
- workspace_utils.py : functions for the workspace not to go to sleep while the model is being trained
- cat_to_name.json : name of the flowers of the dataset
- LICENSE :


## How to train a model
To train a model, run train.py with the desired model architecture (from torchvision.models - by default "vgg13") and the path to the image folder :
```
python train.py --arch vgg13 --data_dir ./flowers
```

Other options are available :
--save_dir &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify where to save your trained model - by default "./checkpoint.pth"  
--npl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify the number of nodes per hidden layer only (neither input layer nor output layer) - by default [1000]  
--epochs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify the number of epochs in the training - by default 10  
--lr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify the learning rate - by default 0.001  
--gpu &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify whether or not the training should be made on the gpu (cuda) - by default True  
--change_class &nbsp; to specify whether you want to keep the standard classifier or create your own - by default True  


## How to make a prediction with a saved model
TO make a prediction, run predict.py  with the image directory :
```
python predict.py --im_dir './flowers/valid/1/image_07094.png'
```

Other options are available :
--checkpoint &nbsp;&nbsp;&nbsp; to specify which saved model is to be used - by default "./checkpoint.pth"  
--topk &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify how much outcomes the model propose and their associated probabilities - by default 5  
--catname &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify the labels corresponding to the data used - by default "cat_to_name.json"  
--gpu &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to specify whether or not the prediction should be made on the gpu (cuda) - by default True  


## Aknowledgements
The data and the workspace_utils.py file were provided by Udacity.
