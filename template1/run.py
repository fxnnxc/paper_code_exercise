"""
Non-parametric
Entropy based MNIST Classifier 
Stage1 : saving MNIST Traing data entropy
"""
import os 
import omegeconf
import torch 
import time 
import argparse
from tqdm import tqdm 
import pickle 
from torch.utils.data import DataLoader

import torchvision 
from torch.utils.data import Dataset
class MNISTWarpper(Dataset):
    def __init__(self, train):
        self.data = torchvision.MNIST()
    
    def __getitem__(self, x):
        return None
    
    
    def __len__(self):
        return None

def compute_entropy(digit_image):
    assert digit_image.size() == (28,28)
    digit_image = digit_image.flatten()
    digit_image = digit_image / digit_image.norm() 
    assert digit_image.sum() == 1
    entropy =  (- digit_image * digit_image.log()).sum() # \sum - p log p
    assert entropy >=0
    return entropy 


# Prepare the experiment 
parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()


CLS_ENTROPY = {
    str(i) : [] for i in range(10)
}

# Run the experiment

train_dataset = MNISTWarpper
train_loader = DataLoader(train_dataset, batch_size=1)
pbar = tqdm(train_loader)
for x,y in pbar:
    CLS_ENTROPY[y] = compute_entropy(x)
    pbar.set_description()
    



# Post process CLS Entropy to make it as tensor and save it
with open('results/cls_entropy.pkl', 'wb') as f:
    pickle.dump(CLS_ENTROPY, f)