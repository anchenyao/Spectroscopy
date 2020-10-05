import pandas as pd
import numpy as np

####FUNCTIONS AND PREPROCESSING####

valence = {'H':1, 'O':2, 'N':3, 'C':4}
valence_inv = {1:'H', 2:'O', 3:'N', 4:'C'}

#max atom size we are working with
max_vertices = 30

#vertex valence remaining (1) + C,O,N,H within 2 edges (4) + atom type (4) + carbon type count (80) + bond type count (17)
per_vertex_dimension = 116


bondTypes = []
carbonBondTypes = []


for a in range(4, 0, -1):
  atom1 = valence_inv[a]
  for b in range(a,0, -1):
    atom2 = valence_inv[b]
    for i in range(1, min(valence[atom1], valence[atom2]) + (atom1 != atom2), 1):
      bondTypes.append((atom1, atom2, i))
      if atom1 == 'C':
        carbonBondTypes.append((atom1, atom2, i))

bondTypes = list(set(bondTypes))
bondTypes.append(('C', 'C', 4))
bondTypes.sort()
carbonBondTypes.append(('C', 'C', 4))
carbonBondTypes.sort()
carbonBondCount = len(carbonBondTypes)

def edge_to_adj(edges):
  return np.array((pd.DataFrame(edges) != 0).astype(int))

def degree(adjM):
  return [sum(adjM[i]) for i in range(len(adjM))]

def twoAway(adjM):
  #returns 2 edges away adjacency matrix
  #adjM^2 - diag matrix with degree of each vertex
  M = np.dot(adjM, adjM)
  degM = degree(adjM)
  for i in range(len(M)):
    M[i][i] -= degM[i]
  return M

#returns a V x 4 matrix, where the columns represent [H, O, N, C] respectively within exactly two edges
def twoAwayPerVertex(edges, vertices):
  twoM = twoAway(edge_to_adj(edges))
  out = []
  for i in range(len(vertices)):
    out_i = [0] * 4
    for j in range(len(vertices)):
      if twoM[i][j] == 1:
        out_i[valence[vertices[j]] - 1] += 1
    out.append(out_i)
  return out


def enumerateCarbonTypes(): #produces list of all carbon types
  cTypes = []
  for a in range(carbonBondCount):
    a_val = carbonBondTypes[a][2]
    valtot_a = a_val
    if valtot_a == 4:
      cTypes.append(tuple(sorted([carbonBondTypes[a][1:3]])))
      continue
    for b in range(a, carbonBondCount, 1):
      b_val = carbonBondTypes[b][2]
      valtot_b = b_val + valtot_a
      if valtot_b == 4:
        cTypes.append(tuple(sorted([carbonBondTypes[a][1:3], carbonBondTypes[b][1:3]])))
        continue
      if valtot_b > 4:
        continue
      for c in range(b, carbonBondCount, 1):
        c_val = carbonBondTypes[c][2]
        valtot_c = c_val + valtot_b
        if valtot_c == 4:
          cTypes.append(tuple(sorted([carbonBondTypes[a][1:3], carbonBondTypes[b][1:3], carbonBondTypes[c][1:3]])))
          continue
        if valtot_c > 4:
          continue
        for d in range(c, carbonBondCount, 1):
          d_val = carbonBondTypes[d][2]
          valtot_d = d_val + valtot_c
          if valtot_d  == 4:
            cTypes.append(tuple(sorted([carbonBondTypes[a][1:3], carbonBondTypes[b][1:3], carbonBondTypes[c][1:3], carbonBondTypes[d][1:3]])))
  cTypes.sort()
  return cTypes

carbonTypes = enumerateCarbonTypes()

#invert to assign index to each carbon type
def carbonIndex(): 
  cTypes = enumerateCarbonTypes()
  outDict = {}
  for i in range(len(cTypes)):
    outDict[cTypes[i]] = i
  return outDict
  
def bondIndex():
  outdict = {}
  for i in range(len(bondTypes)):
    outDict[bondTypes[i]] = i
  return outDict

def countBonds(edges, vertices):
  bindex = bondIndex()
  out = [0] * len(bondTypes)
  for i in range(len(vertices)):
    for j in range(i, len(vertices), 1):
      if edges[i][j] != 0:
        out[bindex[(vertices[i], vertices[j], edges[i][j])]] += 1
  return out

def countCarbons(edges, vertices):
  cindex = carbonIndex()
  out = [0] * len(carbonTypes)
  for i in range(len(vertices)):
    if vertices[i] == 'C':
      bonded = []
      for j in range(len(vertices)):
        if edges[i][j] != 0:
          bonded.append((vertices[j], edges[i][j]))
      bonded = tuple(sorted(bonded))
      out[cindex[bonded]] += 1
  return out

'''maps data to max_v x (3 * max_v + 116) matrix
col [0, 3*max_v - 1] is edge information
col 3*max_v is val remaining
col [3*max_v + 1,3*max_v + 4] is within 2 per-vertex info
col [3*max_v + 5, 3*max_v + 8] is atom type
col [3*max_v + 9, 3*max_v + 88] is carbon type count, same for all vertices
col [3*max_v + 89, 3*max_v + 115] is bond type count, same for all vertices
'''
def featurize(edges, vertices):
  carb_count = countCarbons(edges, vertices)
  bond_count = countBonds(edges, vertices)
  withinTwo = twoAwayPerVertex(edges, vertices)

  out = [[0] * (3 * max_vertices + per_vertex_dimension) for i in range(len(vertices))]
  for i in range(len(vertices)):
    out[i][3 * max_vertices] = valence[vertices[i]]
    out[i][3 * max_vertices + 4 + valence[vertices[i]]] = 1
    for j in range(4): #within two edges
      out[i][3 * max_vertices + 1 + j] = withinTwo[i][j]
    for j in range(80): #carbon types
      out[i][3 * max_vertices + 9 + j] = carb_count[j]
    for j in range(17): #bond types
      out[i][3 * max_vertices + 89 + j] = bond_count[j]


  #map edge matrix to a max_v by 3*max_v binary indicator matrix
  for i in range(len(edges)):
    for j in range(i, len(edges), 1):
      if edges[i][j] != 0:
        out[i][3 * j + edges[i][j] - 1] = 1
        out[i][3* max_vertices] -= edges[i][j]
        out[j][3* max_vertices] -= edges[i][j]
  return out


def generateTrainingPair(edges):
  out = [[0 for i in range(len(edges))] for j in range(len(edges))]
  prob = 0.3 * np.random.uniform(0,1) + 0.7 * np.random.beta(3,3)
  removed = []
  for i in range(len(edges)):
    for j in range(i, len(edges), 1):
      if edges[i][j] != 0:
        if np.random.binomial(1, prob) != 1:
          out[i][j] = edges[i][j]
          out[j][i] = edges[j][i]
        else:
          removed.append((i,j))
  if len(removed) > 0:
    addindex = np.random.randint(len(removed))
    out2 = [[out[i][j] for j in range(len(out))] for i in range(len(out))]
    i_removed = removed[addindex][0]
    j_removed = removed[addindex][1]
    out2[i_removed][j_removed] = edges[i_removed][j_removed]
    out2[j_removed][i_removed] = edges[j_removed][i_removed]
    return (out,out2)
  else:
    return None

def generateTrainingPairsN(edges, N):
  output = []
  for i in range(N):
    trainingPair = generateTrainingPair(edges)
    if trainingPair != None:
      output.append(trainingPair)
  return output

#third dimension, a 1 at index i,j,n means there is a n+1 order bond between i and j
def graph2TensorTo3Tensor(edges):
  out = [[[0 for i in range(3)] for j in range(len(edges))] for j in range(len(edges))]
  for i in range(len(edges)):
    for j in range(len(edges)):
      if edges[i][j] != 0:
        out[i][j][edges[i][j] - 1] = 1
  return out



####CODE BODY####

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

