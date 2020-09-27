import pandas as pd
import numpy as np

####FUNCTIONS AND PREPROCESSING####

valence = {'H':1, 'O':2, 'N':3, 'C':4}
valence_inv = {1:'H', 2:'O', 3:'N', 4:'C'}

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

#returns a V x 4 matrix, where the columns represent [H, O, N, C] respectively within two edges
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
  out = [0] * len *
  for i in range(len(vertices)):
    if vertices[i] == 'C':
      bonded = []
      for j in range(len(vertices)):
        if edges[i][j] != 0:
          bonded.append((vertices[j], edges[i][j]))
      bonded = tuple(sorted(bonded))
      out[cindex[bonded]] += 1
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
