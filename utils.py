import os
import sys
import time
import math
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import *
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

first_run = True

def get_dataset(dataset, train_val_split = 0.9):
    '''
    This function downloads the dataset chosen by the user, performs transformations on the dataset and outputs datagenerator objects for train/val/test sets
    '''

    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if dataset == 'cifar10':
        train_data = CIFAR10(root='./data', train = True, download = first_run, transform = transform)
        test_data  = CIFAR10(root='./data', train = False, download = first_run, transform = transform)

    if dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train = True, download = first_run, transform = transform)
        test_data  = CIFAR100(root='./data', train = False, download = first_run, transform = transform)
        
    train_size = int(train_val_split * len(train_data))
    val_size = len(train_data) - train_size


    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    return train_data, val_data, test_data


def get_dataloader(dataset, batch_size, train_val_split = 0.9, num_workers = 1):
    '''
    This function generates dataloader objects using get_dataset() function defined
    '''
    
    if dataset not in ['cifar10', 'cifar100']:
        raise ValueError("Wrong dataset. Select one from ['cifar10', 'cifar100']")
        
    train_set, val_set, test_set = get_dataset(dataset, train_val_split)
    trainloader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    valloader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)
    testloader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)

    return trainloader, valloader, testloader

class Padam(Optimizer):
    '''
    This class defines the proposed Padam optimizer and contains the step function which performs optimization step during training process. Input parameters for the constructor are defined as follows:
    lr: learning rate 
    betas: beta values (beta1, beta2)
    eps: epsilon value 
    weight_decay: weight decay value
    amsgrad_flag: flag for the use of amsgrad 
    partial: p hyperparameter
    '''
    
    def __init__(self, params, 
                 lr = 0.1, 
                 betas = (0.9, 0.999), 
                 eps = 1e-8, 
                 weight_decay = 0, 
                 amsgrad_flag = True, 
                 partial = 1/4):
        
        if betas[0] >= 1.0 or betas[0] < 0.0:
            raise ValueError("Beta0 parameter is not in range [0, 1)")
            
        if betas[0] >= 1.0 or betas[1] < 0.0:
            raise ValueError("Beta1 parameter is not in range [0, 1)")
            
        super(Padam, self).__init__(params, dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, amsgrad_flag = amsgrad_flag, partial = partial))

    def step(self, closure = None):
        
        loss = None
        
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            for p in group['params']:
             
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError('Sparse Gradients are not supported.')
                    
                amsgrad_flag = group['amsgrad_flag']
                partial = group['partial']

                curr_state = self.state[p]
                
                #Initialize the state of optimizer
                if len(curr_state) == 0:
                    
                    curr_state['step'] = 0
                    curr_state['exp_avg'] = torch.zeros_like(p.data) #Defines the exponential moving average of grad values
                    curr_state['exp_avg_square'] = torch.zeros_like(p.data) #Defines the exponential moving average of squared grad values
                    
                    if amsgrad_flag:
                        curr_state['max_exp_avg_square'] = torch.zeros_like(p.data) #Defines the max value of all exponential moving averages of squared grad values

                exp_avg, exp_avg_square = curr_state['exp_avg'], curr_state['exp_avg_square']
                
                if amsgrad_flag:
                    max_exp_avg_square = curr_state['max_exp_avg_square']
                    
                beta1, beta2 = group['betas']

                curr_state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha = group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1) #Decays the first and second order moments of grad values
                exp_avg_square.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2) #Decays the first and second order moments of squared grad values
                
                if amsgrad_flag: #Check whether to use amsgrad or not
                    torch.max(max_exp_avg_square, exp_avg_square, out = max_exp_avg_square) #Compute the max of 2nd moment running averages till current iteration
                    denominator = max_exp_avg_square.sqrt().add_(group['eps']) #Compute the denominator term using the max operator for normalizing running average
                    
                else:
                    denominator = exp_avg_square.sqrt().add_(group['eps']) #If amsgrad is not used, compute denominator using exponential average of squared grad values

                bias1_correction = 1 - beta1 ** curr_state['step'] #Defines the correction for first bias term
                bias2_correction = 1 - beta2 ** curr_state['step'] #Defines the correction for second bias term
                
                step_size = group['lr'] * math.sqrt(bias2_correction) / bias1_correction #Compute the step size

                p.data.addcdiv_(exp_avg, denominator**(partial*2), value = -step_size)
                
        return loss