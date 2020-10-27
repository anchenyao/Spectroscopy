import pandas as pd
import numpy as np
import scfunctions as sc

max_vertices = 40

#MAKE SURE TO FILTER BASED ON MAX_VERTICES
molData = np.load('data2020-10-14valencetrimmed.npy', allow_pickle = True)
#np.save('data2020-10-14valencetrimmed.npy', trimBadValence(molData))
molData = sc.trimVertexCount(molData, max_vertices)
molData = np.array([[molData[i][0], sc.graph2TensorTo3Tensor(np.array(molData[i][1]).astype(int))] for i in range(len(molData))])

#HERE
#TESTING

'''
import sys
np.set_printoptions(threshold=sys.maxsize)

mData2 = np.load('data2020-10-14valencetrimmed.npy')
dataPerMol = 1
testing = sc.generateTrainingPairsN(sc.graph2TensorTo3Tensor(mData2[0][1]), mData2[0][0], dataPerMol, max_vertices)[0]
feats = np.array(testing[0])
curedges = feats[:,:3*max_vertices].sum()
probs = testing[1]
newedges = (probs > 0).sum()/2
totedges = np.array(mData2[0][1]).sum()/2

#MISSING EDGES?????????
'''
startind = 0
endind = 4000
molSubset = molData[startind:endind]
trainingData = [] #[molecule][datapointnumber][input or proboutput]
dataPerMol = 50
for i in range(len(molSubset)):
  if i% 10 == 0:
    print(i)
  trainingData.append(sc.generateTrainingPairsN(molSubset[i][1], molSubset[i][0], dataPerMol, max_vertices))
  if i% 100 == 0:
    np.save('training0to4000mols50perData2020-10-27.npy', np.array(trainingData))

np.save('training0to4000mols50perData2020-10-27.npy', np.array(trainingData))


































#NN
'''GRU implementation found online
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=5, model_type="GRU"):
    
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE
'''

