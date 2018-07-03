import argparse
import helper
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

parser = argparse.ArgumentParser()


parser.add_argument('--gpu', action = 'store_true', 
                    dest = 'gpu', 
                    default = False, 
                    help='Use GPU if --gpu')

parser.add_argument('--top_k', action='store',
                    dest = 'top_k',
                    type = int, 
                    default = 1,
                    help = 'Top K possibilities')

parser.add_argument('--img', action = 'store',
                    dest = 'img',
                    type = str, 
                    default = 'Sample_image.jpg',
                    help = 'Store Img Name')

parser.add_argument('--category_names', action = 'store',
                    dest = 'cat',
                    type = str, 
                    default = 'cat_to_name.json',
                    help = 'json name to map categories')

parser.add_argument('--st', action = 'store_true',
                    default = False,
                    dest = 'start',
                    help = '--st to start predicting')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_units',
                    type = int, 
                    default = 512,
                    help = 'Number of hidden units')

parser.add_argument('--arch', action = 'store',
                    dest = 'arch',
                    type = str, 
                    default = 'densenet',
                    help = 'PreTrained Model Architecture, densenet or vgg')

results = parser.parse_args()
print('---------Parameters----------')
print('gpu              = {!r}'.format(results.gpu))
print('img              = {!r}'.format(results.img))
print('top_k            = {!r}'.format(results.top_k))
print('cat              = {!r}'.format(results.cat))
print('start            = {!r}'.format(results.start))

print('-----------------------------')

if results.start == True:
    model, class_labels = helper.load_saved_model()
    cat_to_name, label_order = helper.load_json(results.cat)
    ps, labels, index = helper.predict(results.img, model, results.top_k, cat_to_name, class_labels, results.gpu)
    print("------------------Prediction------------------")
    for i in range(len(ps)):
        print("The probability of the flower to be {} is {:.2f} %.".format(labels[i], ps[i] * 100))
    
    
    