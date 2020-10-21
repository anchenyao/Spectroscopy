import pandas as pd
import numpy as np
import scfunctions as sc
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

max_vertices = 40

tdata = np.load('training0to40Data1000per2020-10-20.npy', allow_pickle = True) #FIX OUTPUT DIMENSIONS
tlist = []

for i in range(len(tdata)):
  for j in range(len(tdata[i])):
    tlist.append(tdata[i][j])

tlist = tlist[:int(len(tlist)/64) * 64]
tlist = [[torch.FloatTensor(tlist[i][0]),torch.FloatTensor(tlist[i][1])] for i in range(len(tlist))]
trainloader = torch.utils.data.DataLoader(tlist, batch_size=64, shuffle=True)
#####################################################################################################################
#LEARNING TO LEARN
stlist = tlist#[:128]
stlist = [[torch.FloatTensor(stlist[i][0]),sc.padZeros(torch.FloatTensor(stlist[i][1]),max_vertices, max_vertices)] for i in range(len(stlist))]
inp = stlist[0][0]
oot = stlist[0][1]

#input/output dims
idim = inp.shape[0] * inp.shape[1]
odim = oot.shape[0] * oot.shape[1] * oot.shape[2]

#turn in/out into 1d vectors
stlist = [[stlist[i][0].view(idim),stlist[i][1].view(odim)] for i in range(len(stlist))]
trainloader = torch.utils.data.DataLoader(stlist, batch_size=64, shuffle=True)

#layers
numLinLayers = 10

layerlist = [nn.Linear(idim, 256), nn.ReLU()]
for i in range(numLinLayers):
  layerlist.append(nn.Linear(256, 256))
  layerlist.append(nn.ReLU())
layerlist.append(nn.Linear(256, odim))
layerlist.append(nn.Softmax())


#NN
model = nn.Sequential(*layerlist)
criterion = nn.BCELoss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
epochs = 100
for e in range(epochs):
  print('epoch = ' + str(e))
  running_loss = 0
  for feats, probs in trainloader:
    feats = feats.view(feats.shape[0], -1)
    optimizer.zero_grad()
    output = model(feats)
    loss = criterion(output, probs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  print(f"Training loss: {running_loss/len(trainloader)}")


'''
class Network(nn.Module):
  def __init__(self):
    super().__init__()
    
    # Inputs to hidden layer linear transformation
    self.hidden = nn.Linear(784, 256)
    # Output layer, 10 units - one for each digit
    self.output = nn.Linear(256, 10)
    
    # Define sigmoid activation and softmax output 
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax(dim=1)
      
  def forward(self, x):
    # Pass the input tensor through each of our operations
    x = self.hidden(x)
    x = self.sigmoid(x)
    x = self.output(x)
    x = self.softmax(x)
    
    return x
'''