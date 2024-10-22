import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.conv1=nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1=nn.Linear(10*10*128, 512)
        self.fc2=nn.Linear(512, 256)
        self.fc2=nn.Linear(256, action_size)

    def forward(self, state):
       x=F.relu(self.bn1( self.conv1(state)))
       x=F.relu(self.bn2( self.conv2(x)))
       x=F.relu(self.bn3( self.conv3(x)))
       x=F.relu(self.bn4( self.conv4(x)))
       x=x.view(x.size(0), -1)
       x=F.relu(self.fc1(x))
       x=F.relu(self.fc2(x))
       return x

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env=gym.make("MsPacmanDeterministic-v0", full_action_space=False)

state_shape=env.observation_space.shape
number_actions=env.action_space.n

print(state_shape)
print(number_actions)


learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    frame=Image.fromarray(frame)
    preprocess=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)


