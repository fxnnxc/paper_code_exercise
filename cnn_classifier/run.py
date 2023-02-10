
from datasets import get_datasets
from models import CNN
from omegaconf import OmegaConf
import random 
import numpy as np 
import torch 
import argparse
import time 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--data-path", default='untracked')
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()
flags = OmegaConf.load("config.yaml")
for key in vars(args):
    setattr(flags, key, getattr(args, key))
    
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

flags.save_path = "results"
flags.start_time = time.time()

import os 
if not os.path.exists(flags.save_path):
    os.makedirs(f"{flags.save_path}")
if not os.path.exists(flags.data_path):
    os.makedirs(flags.data_path)


in_channels = 1 if "mnist" in flags.data else 3  # 
model = CNN(in_channels=in_channels)
train_dataset, valid_dataset = get_datasets(flags.data, data_path=flags.data_path)
train_loader =  DataLoader(train_dataset, batch_size=flags.batch_size) 
valid_loader = DataLoader(valid_dataset, batch_size=flags.batch_size) 
 
model.to(flags.device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=flags.learning_rate)
lr_scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.8**epoch)

writer = SummaryWriter(os.path.join(flags.save_path, "runs"))
count = 0
best_performance = 0
running_loss = 0

flags.start_time = time.time()
flags.results = {}
optimizer.zero_grad()
for epoch in range(flags.epochs):
    # training
    model.train()
    pbar = tqdm((train_loader))
    for batch_idx, (x,y) in enumerate(pbar):
        x,y = x.to(flags.device), y.to(flags.device)
        y_hat = model.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        running_loss += loss.item()
        count += 1
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        writer.add_scalar(f"RunningLoss(CE)", running_loss/count, count)      
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))         
        pbar.set_description(f"{flags.save_path} E:({epoch/flags.epochs:.2f}) D:({duration})]")    

    # eval
    model.eval()
    eq = 0
    for k, (x, y) in (enumerate(valid_loader)):
        x = x.to(flags.device)
        y = y.to(flags.device)
        y_hat = model.forward(x).argmax(dim=-1)
        eq += (y == y_hat).sum()
    
    current_performance = eq/len(valid_dataset)
    writer.add_scalar(f"Valid-ACC", current_performance, epoch)
    if current_performance > best_performance:
        best_performance = current_performance
        torch.save(model, os.path.join(flags.save_path, f"models/model_best.pt"))
    if epoch % flags.save_epochs==0 or epoch == flags.epochs -1:
        torch.save(model, os.path.join(flags.save_path, f"models/model_{epoch}.pt"))
    
    path = f"{flags.save_path}/checkpoint_{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), f"{path}/model.pt")
    OmegaConf.save(flags, f"{path}/config.yaml")
    flags.result.append({'epoch': epoch, 
                            'performance':current_performance.item(), 
                            'lr':lr_scheduler.get_last_lr()[0]})

    lr_scheduler.step()
    