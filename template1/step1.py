"""
Non-parametric
Entropy based MNIST Classifier 
Stage1 : saving MNIST Traing data entropy
"""
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader
class MNISTWarpper(Dataset):
    def __init__(self, root, train, transform):
        self.data = torchvision.datasets.MNIST(root=root, train=train, transform=transform)
    
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)

def compute_entropy(digit_image):
    assert digit_image.size() == (28,28)
    digit_image = digit_image.flatten()
    digit_image = digit_image / digit_image.norm() 
    assert digit_image.sum() == 1
    entropy =  (- digit_image * digit_image.log()).sum() # \sum - p log p
    assert entropy >=0
    return entropy 


import os 
import time 
import pickle 
import random
import numpy as np 
import argparse
import datetime 
from tqdm import tqdm 
from omegaconf import OmegaConf

# ==== 🔖 Argument Setting ====
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config.yaml')
parser.add_argument("--save-name", default='config.yaml')
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

flags = OmegaConf.load(args.config)
for key in vars(args):
    setattr(flags, key, getattr(args, key))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

date = datetime.datetime.now().strftime(format="%Y-%m-%d--%H-%M-%S")
flags.save_dir = f"results/{flags.seed}_{date}"
flags.start_time = time.time()

if not os.path.exists(flags.save_dir):
    os.makedirs(flags.save_dir)
OmegaConf.save(flags, f'{flags.save_dir}/config.yaml')


# ==== 🔖 Running the Experiment ====
CLS_ENTROPY = {str(i) : [] for i in range(10)} # holder for the entropy for class samples

train_dataset = MNISTWarpper(flags.data_path, train=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=1)
pbar = tqdm(enumerate(train_loader))
for i,(x,y) in pbar:
    entropy = compute_entropy(x)
    CLS_ENTROPY[y].append(compute_entropy(x))
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))
    pbar.set_description(f"[INFO] {flags.save_dir}| N:({i:.2E}) P:({i / len(train_dataset)*100:.2f}%) D:({duration}) | out-dist eval :")


# Savae the result 
with open(f'{flags.save_dir}/cls_entropy.pkl', 'wb') as f:
    pickle.dump(CLS_ENTROPY, f)