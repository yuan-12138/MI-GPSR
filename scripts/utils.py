#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:26:43 2020

@author: yuanbi
"""
from Dataset_loader import Dataset_loader
from torch.utils.data.dataloader import DataLoader

import itertools


BATCH_SIZE = 128
NUM_WORKERS = 2


def prepare_data(length_reduced, poses, length,NUM_DEMO,files,demo_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    training_set = Dataset_loader(list(itertools.combinations(list(range(0,length_reduced.sum())),2)), length,
                                 length_reduced, poses,NUM_DEMO,files,demo_path)
	# num_workers denotes how many subprocesses to use for data loading
    trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('trainloader size:', len(training_set))   
    
    return trainloader


def prepare_test_data(length_reduced, poses, length,NUM_DEMO,files,demo_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    test_set = Dataset_loader(list(itertools.combinations(list(range(0,length_reduced.sum())),2)), length, 
                                 length_reduced, poses,NUM_DEMO,files,demo_path)
	# num_workers denotes how many subprocesses to use for data loading
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('testloader size:', len(test_set))   
    
    return testloader

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
