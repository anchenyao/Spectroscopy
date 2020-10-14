import numpy as np
import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


my_sdf_file = 'compounds.sdf'

suppl = Chem.SDMolSupplier(my_sdf_file)

#pairs of vertices, edges
anumDict = {1:'H', 6:'C', 7:'N', 8:'O'}
output = []
ct = 0
for mol in suppl:
  v = []
  for atom in mol.GetAtoms():
    try:
      v.append(anumDict[atom.GetAtomicNum()])
    except:
      v = []
      break
  if v != []:
    e = [[0 for j in range(len(v))] for i in range(len(v))]
    intbonds = 1
    for b in mol.GetBonds():
      sid = b.GetBeginAtomIdx()
      eid = b.GetEndAtomIdx()
      bondOrd = b.GetBondTypeAsDouble()
      if int(bondOrd) == bondOrd:
        e[sid][eid] = b.GetBondTypeAsDouble() 
        e[eid][sid] = b.GetBondTypeAsDouble() 
      else:
        intbonds = 0
        break
    if intbonds:
      output.append([v,e])
  ct +=1
  if ct % 100 == 0:
    print(ct)

np.save('data2020-10-14.npy', np.array(output))


'''
def pcp_convert(d):
  vertices = []
  for i in range(len(d['atoms'])):
    vertices.append(d['atoms'][i]['element'])
  edges = [[0 for j in range(len(vertices))] for i in range(len(vertices))]
  for i in range(len(d['bonds'])):
    a1 = d['bonds'][i]['aid1'] - 1
    a2 = d['bonds'][i]['aid2'] - 1
    order = d['bonds'][i]['order']
    edges[a1][a2] = order
    edges[a2][a1] = order
  return (vertices, edges)

ids = pcp.get_compounds('COHN', 'smiles', searchtype='superstructure', listkey_count=100)

data = []
for cid in ids:
  c = pcp.Compound.from_cid(cid.cid).to_dict(properties=['atoms', 'bonds'])
  c_data = pcp_convert(c)
  if 'C' in c_data[0]:
    data.append(c_data)
'''





'''
idlist = []
for j in range(1000):
  print(j)
  ids = pcp.get_compounds('COHN', 'smiles', searchtype='superstructure', listkey_count=100, listkey_start = 100*j)
  for i in range(len(ids)):
    idlist.append(ids[i])
data = []
for cid in idlist:
  c = pcp.Compound.from_cid(cid.cid).to_dict(properties=['atoms', 'bonds'])
  c_data = pcp_convert(c)
  if 'C' in c_data[0]:
    data.append(c_data)
'''

