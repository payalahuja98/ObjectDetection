import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Alexnet(nn.Module):
  def __init__(self):
      super(Alexnet, self).__init__()
      self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
      self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
      self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
      self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
      self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
      self.pool = nn.MaxPool2d(kernel_size=2)
      self.fc1 = nn.Linear(256 * 7 * 7, 4096)
      self.fc2 = nn.Linear(4096, 4096)
      self.fc3 = nn.Linear(4096, NUM_CLASSES)
      self.drop = nn.Dropout()

      self.bn1 = nn.BatchNorm2d(64)
      self.bn2 = nn.BatchNorm2d(192)
      self.bn3 = nn.BatchNorm2d(384)
      self.bn4 = nn.BatchNorm2d(256)

      # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
  def forward(self, x): 
      x = self.bn1(self.conv1(x))
      x = self.pool(F.relu(x))     
      x = self.bn2(self.conv2(x))
      x = self.pool(F.relu(x))
      x = self.bn3(self.conv3(x))
      x = self.pool(F.relu(x))
      x = self.bn4(self.conv4(x))
      x = self.pool(F.relu(x))
      x = self.pool(F.relu(self.conv5(x)))
      x = x.view(x.size()[0], 256 * 7 * 7)
      x = self.drop(x)
      x = F.relu(self.fc1(x))
      x = self.drop(x)
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

      
class Classifier(nn.Module):
  # TODO: implement me
  def __init__(self):
      super(Classifier, self).__init__()
      # self.conv0 = nn.Conv2d(3, 256, 4)
      self.conv1 = nn.Conv2d(3, 128, 5)#in_channel, out_chafeatunnel, kernel
      self.conv2 = nn.Conv2d(128, 64, 5)#in_channel, out_chafeatunnel, kernel
      self.conv3 = nn.Conv2d(64, 32, 3)
      self.conv4 = nn.Conv2d(32, 16, 3)
      self.pool = nn.MaxPool2d(2, 2)
      self.fc1 = nn.Linear(16* 11 * 11, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, NUM_CLASSES)
      self.drop = nn.Dropout(0.2)
      
  
  def forward(self, x):

      # x = self.pool(F.relu(self.conv0(x)))

      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.pool(F.relu(self.conv4(x)))
      # print('x size is', x.size())
      x = x.view(x.size()[0], 16* 11 * 11)
      x = F.relu(self.fc1(x))
      x = self.drop(x)
      x = F.relu(self.fc2(x))
      x = self.fc3(x)

      # x = self.pool(F.relu(self.conv1(x)))
      # x = self.pool(F.relu(self.conv2(x)))
      # x = self.pool(F.relu(self.conv3(x)))
      # x = x.view(x.size()[0], 16 * 26 * 26)
      # x = F.relu(self.fc1(x))
      # x = F.relu(self.fc2(x))
      # x = self.fc3(x)
      return x


