import torch 
import torch.nn as nn 

# def get_model(name):
#     model = {
#             "mnist":MnistCNN(),
#             "cifar10":MnistCNN(),
#             "cifar100":MnistCNN(),
#             "fasion_mnist":MnistCNN(),
#         }[name]
#     return model 



class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels,16,5,2,4),   nn.ReLU())
        self.cnn2 = nn.Sequential(nn.BatchNorm2d(16), nn.Conv2d(16,32,3,2,1),  nn.ReLU())
        self.cnn3 = nn.Sequential(nn.BatchNorm2d(32), nn.Conv2d(32,64,3,2,1),  nn.ReLU())
        self.cnn4 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64,64,3,2,1),  nn.ReLU())
        self.flatten = nn.Flatten()
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.flatten(x)
        x = self.out(x)
        return x

    
# class CIFAR10CNN():
#     def __init__(self):
#         pass 
#     def forward(self, x):
#         pass 
        

# class CIFAR100CNN():
#     def __init__(self):
#         pass 
#     def forward(self, x):
#         pass 
    