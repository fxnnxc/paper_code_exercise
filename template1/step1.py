"""
Non-parametric
Entropy based MNIST Classifier 
Stage1 : saving MNIST Traing data entropy
"""


import os 
import torch 
import time 
from tqdm import tqdm 
import pickle 
from torch.utils.data import DataLoader

import torchvision 
from torch.utils.data import Dataset
class MNISTWarpper(Dataset):
    def __init__(self, train=True):
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToTensor()
        ])
        self.data = torchvision.datasets.MNIST(train=train, root='untracked', download=True, transform=transform)
    
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)

def compute_entropy(digit_image):
    assert digit_image.size() == (1, 28,28)
    digit_image = digit_image.flatten()
    digit_image = digit_image / digit_image.sum()
    print(digit_image.sum())
    assert abs(digit_image.sum() - 1.0) <1e-14
    entropy =  (- digit_image * torch.nan_to_num(digit_image.log())).sum() # \sum - p log p
    assert entropy >=0
    return entropy 


from omegaconf import OmegaConf
import argparse
# Prepare the experiment 
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config.yaml')
parser.add_argument("--learning-rate", type=float, default=1e-3)
args = parser.parse_args()

# learning_rate = 1e-3
flags = OmegaConf.load(args.config)
flags.learning_rate = args.learning_rate
save_dir = f"results/exp1"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
OmegaConf.save(flags, f'{save_dir}/config.yaml')

CLS_ENTROPY = {
    str(i) : [] for i in range(10)
}


# Run the experiment

train_dataset = MNISTWarpper(train=True)
train_loader = DataLoader(train_dataset, batch_size=1)
pbar = tqdm(train_loader)
for x,y in pbar:
    x = x.squeeze(0)
    y = y.item()
    CLS_ENTROPY[y] = compute_entropy(x)
    pbar.set_description()
    



# Post process CLS Entropy to make it as tensor and save it
with open(f'{save_dir}/cls_entropy.pkl', 'wb') as f:
    pickle.dump(CLS_ENTROPY, f)