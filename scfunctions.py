import pandas as pd
import numpy as np
import torch
from datetime import date

####FUNCTIONS AND PREPROCESSING####

valence = {'H':1, 'O':2, 'N':3, 'C':4}
valence_inv = {1:'H', 2:'O', 3:'N', 4:'C'}

#vertex valence remaining (1) + C,O,N,H within 2 edges (4) + atom type (4) + carbon type count (80) + bond type count (17)
per_vertex_dimension = 116


bondTypes = []
carbonBondTypes = []


for a in range(4, 0, -1):
  atom1 = valence_inv[a]
  for b in range(a,0, -1):
    atom2 = valence_inv[b]
    for i in range(1, min(valence[atom1], valence[atom2]) + (atom1 != atom2), 1):
      bondTypes.append((min(atom1, atom2), max(atom1, atom2) , i))
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
  outDict = {}
  for i in range(len(bondTypes)):
    outDict[bondTypes[i]] = i
  return outDict

def countBonds(edges, vertices):
  bindex = bondIndex()
  out = [0] * len(bondTypes)
  for i in range(len(vertices)):
    for j in range(i, len(vertices), 1):
      if edges[i][j] != 0:
        out[bindex[(min(vertices[i], vertices[j]), max(vertices[i], vertices[j]), edges[i][j])]] += 1
        #out[bindex[(vertices[i], vertices[j], edges[i][j])]] += 1
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
      try: # in case we are looking at subgraph, carbon type may not be complete
        out[cindex[bonded]] += 1
      except:
        pass
  return out


'''maps data to max_v x (3 * max_v + 116) matrix
col [0, 3*max_v - 1] is edge information
col 3*max_v is val remaining
col [3*max_v + 1,3*max_v + 4] is within 2 per-vertex info (remaining)
col [3*max_v + 5, 3*max_v + 8] is atom type
col [3*max_v + 9, 3*max_v + 88] is carbon type count, same for all vertices (remaining)
col [3*max_v + 89, 3*max_v + 115] is bond type count, same for all vertices (remaining)
'''
def featurize(edges, vertices, subedges, max_vertices): 
  carb_count = countCarbons(edges, vertices)
  bond_count = countBonds(edges, vertices)
  withinTwo = twoAwayPerVertex(edges, vertices)
 
  carb_countSub = countCarbons(subedges, vertices)
  bond_countSub = countBonds(subedges, vertices) 
  withinTwoSub = twoAwayPerVertex(subedges, vertices)


  out = [[0] * (3 * max_vertices + per_vertex_dimension) for i in range(max_vertices)]
  for i in range(len(vertices)):
    out[i][3 * max_vertices] = valence[vertices[i]] - sum(subedges[i])
    out[i][3 * max_vertices + 4 + valence[vertices[i]]] = 1
    for j in range(4): #within two edges
      out[i][3 * max_vertices + 1 + j] = withinTwo[i][j] - withinTwoSub[i][j]
    for j in range(80): #carbon types
      out[i][3 * max_vertices + 9 + j] = carb_count[j] - carb_countSub[j]
    for j in range(17): #bond types
      out[i][3 * max_vertices + 89 + j] = bond_count[j] - bond_countSub[j]


  #map edge matrix to a max_v by 3*max_v binary indicator matrix
  for i in range(len(subedges)):
    for j in range(i, len(subedges), 1):
      if subedges[i][j] != 0:
        out[i][3 * j + subedges[i][j] - 1] = 1
  return out


def generateTrainingPair(edges, vertices, max_vertices): #FIX TURN INTO PROBABILITIES
  out = [[[0]*3 for i in range(len(edges))] for j in range(len(edges))]
  prob = 0.3 * np.random.uniform(0,1) + 0.7 * np.random.beta(3,3)
  removed = []
  for i in range(len(edges)):
    for j in range(i, len(edges), 1):
      for k in range(3):
        if edges[i][j][k] != 0:
          if np.random.binomial(1, prob) != 1:
            out[i][j][k] = 1
          else:
            removed.append((i,j,k))

  if len(removed) > 0:
    out2 = [[[0 for k in range(3)] for j in range(len(out))] for i in range(len(out))]
    for a in range(len(removed)):
      out2[removed[a][0]][removed[a][1]][removed[a][2]] = 1.0/len(removed) #UPPER TRIANGULAR REPRESENTATION
    return (featurize(graph3TensorTo2Tensor(edges), vertices, graph3TensorTo2Tensor(out), max_vertices),out2)
  else:
    return None

def generateTrainingPairsN(edges, vertices, N, max_vertices):
  output = []
  for i in range(N):
    trainingPair = generateTrainingPair(edges, vertices, max_vertices)
    if trainingPair != None:
      output.append(trainingPair)
  return output

#third dimension, a 1 at index i,j,n means there is a n+1 order bond between i and j
def graph2TensorTo3Tensor(edges):
  out = [[[0 for i in range(3)] for j in range(len(edges))] for j in range(len(edges))]
  for i in range(len(edges)):
    for j in range(len(edges)):
      if edges[i][j] != 0:
        out[i][j][int(edges[i][j] - 1)] = 1
  return out

def graph3TensorTo2Tensor(edges):
  out = [[0 for i in range(len(edges))] for j in range(len(edges))]
  for i in range(len(edges)):
    for j in range(len(edges)):
      for k in range(3):
        if edges[i][j][k] != 0:
          out[i][j] = k+1
  return out

def trimBadValence(mols): #mols is (v, e), where e is 2Tensor, runs in O(mols^2), can make O(mols) if necessary by using pointers on 'bad'
  bad = []
  out = []
  for i in range(len(mols)):
    vals = [valence[mols[i][0][j]] for j in range(len(mols[i][0]))]
    edge_val_sum = list(pd.DataFrame(mols[i][1]).sum())
    if vals != edge_val_sum:
      bad.append(i)
  for i in range(len(mols)):
    if i not in bad:
      out.append(mols[i])
  return out

def trimVertexCount(mols, maxv):
  out = []
  for i in range(len(mols)):
    if len(mols[i][0]) <= maxv:
      out.append(mols[i])
  return out

def printMatrix(twoTensor):
  for i in range(len(twoTensor)):
    print(np.array(twoTensor[i]).astype(int))

def matrixDiffs(m1, m2):
  m1count = (np.array(m1) != 0).sum()
  m2count = (np.array(m2) != 0).sum()
  diffM = (np.array(m1) != np.array(m2))
  sameM = (np.array(m1) == np.array(m2))
  return diffM.sum(), m1count, m2count


def generateTrainingPairsFull(molStart, molEnd, dataPerMol, max_vertices, date = date.today()):
  molData = np.load('data2020-10-14valencetrimmed.npy')
  molData = trimVertexCount(molData, max_vertices)
  molData = np.array([[molData[i][0], graph2TensorTo3Tensor(np.array(molData[i][1]).astype(int))] for i in range(len(molData))])
  molSubset = molData[molStart:molEnd]
  trainingData = [] #[molecule][datapointnumber][input or proboutput]
  for i in range(len(molSubset)):
    print(i)
    trainingData.append(generateTrainingPairsN(molSubset[i][1], molSubset[i][0], dataPerMol, max_vertices))
    if i%10 == 0:
      np.save('training{}to{}Data{}per{}.npy'.format(str(molStart), str(molEnd), str(dataPerMol), str(date.today())), np.array(trainingData))
  np.save('training{}to{}Data{}per{}.npy'.format(str(molStart), str(molEnd), str(dataPerMol), str(date.today())), np.array(trainingData))

def verifyTrainingPair(feats, nextprobs):
  preedges = np.zeros((len(nextprobs),len(nextprobs)))
  nextedges3T = (np.array(nextprobs) != 0)
  nextedges = graph3TensorTo2Tensor(nextedges3T)
  for i in range(len(nextprobs)):
    for j in range(len(nextprobs)):
      for k in range(3):
        if feats[i][3*j+k] != 0:
          preedges[i][j] = k+1
  counts = matrixDiffs(preedges, nextedges)
  return [counts[0] == (counts[1] + counts[2]), counts]

def verifyTrainingPairSet(tdata):
  correct = 0
  wrong = []
  for i in range(len(tdata)):
    for j in range(len(tdata[i])):
      if verifyTrainingPair(tdata[i][j][0],tdata[i][j][1]):
        correct += 1
      else:
        wrong.append(tdata[i][j])
  return correct, len(tdata) * len(tdata[0]) -correct, wrong

def padZeros(mtx, outx, outy):
  output = np.zeros([outx,outy,3])
  output[:mtx.shape[0],:mtx.shape[1],:3] = mtx
  return torch.FloatTensor(output)

#sum of probabilities of correct edges
def probabilityCorrect(modeloutputprobs, probs):
  correctedges = probs > 0
  correctprobs = correctedges * modeloutputprobs
  return correctprobs.sum()

def trimLowProbabilities(probs, iters, cutoff = 1/(40*40*3)):
  newprobs = (probs > cutoff) * probs
  newprobs /= newprobs.sum()
  for i in range(1,iters):
    newprobs = (newprobs > cutoff) * newprobs
    newprobs /= newprobs.sum()
  return newprobs


