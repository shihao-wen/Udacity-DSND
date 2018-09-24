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

parser.add_argument('--epochs', action='store',
                    dest = 'epochs',
                    type = int, 
                    default = 1,
                    help = 'Number of epochs')

parser.add_argument('--arch', action = 'store',
                    dest = 'arch',
                    type = str, 
                    default = 'densenet',
                    help = 'PreTrained Model Architecture, densenet or vgg')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'learning_rate',
                    type = float, 
                    default = 0.001,
                    help = 'Learning rate')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_units',
                    type = int, 
                    default = 512,
                    help = 'Number of hidden units')

parser.add_argument('--st', action = 'store_true',
                    default = False,
                    dest = 'start',
                    help = '-st to start training')

results = parser.parse_args()
print('---------Parameters----------')
print('gpu              = {!r}'.format(results.gpu))
print('epoch(s)         = {!r}'.format(results.epochs))
print('arch             = {!r}'.format(results.arch))
print('learning_rate    = {!r}'.format(results.learning_rate))
print('hidden_units     = {!r}'.format(results.hidden_units))
print('start            = {!r}'.format(results.start))
print('-----------------------------')

if results.start == True:
    class_labels, trainloader, testloader, validloader = helper.load_img()
    model = helper.load_pretrained_model(results.arch, results.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = results.learning_rate)
    helper.train_model(model, results.learning_rate, criterion, trainloader, validloader, results.epochs, results.gpu)
    helper.test_model(model, testloader, results.gpu)
    model.to('cpu')
    
    # Save Checkpoint for predection
    helper.save_checkpoint({
                'arch': results.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hidden_units': results.hidden_units,
                'class_labels': class_labels
                })
    print('Checkpoint has been saved.')
    
