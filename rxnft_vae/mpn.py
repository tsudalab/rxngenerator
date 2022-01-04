import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from reaction_utils import get_mol_from_smiles
from nnutils import *






ELEM_LIST = ['C', 'N', 'O', 'S' 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
BTYPE_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
ATOM_DIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_DIM = 5 + 6
MAX_NB = 6


def onek_encoding_unk(x, allowable_set):
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: int(x==s), allowable_set))


def atom_features(atom):
	feat = onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) \
	+ onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]) + onek_encoding_unk(atom.GetChiralTag(), [0,1,2,3]) + [int(atom.GetIsAromatic())]
	return torch.Tensor(feat)

def bond_features(bond):
	bt = bond.GetBondType()
	stereo = int(bond.GetStereo())
	feat = onek_encoding_unk(bt, BTYPE_LIST) + [int(bond.IsInRing())] + onek_encoding_unk(stereo, [0,1,2,3,4,5])
	return torch.Tensor(feat)

def mol2graph(mol_batch):
	padding = torch.zeros(ATOM_DIM + BOND_DIM)
	fatoms = []
	fbonds = [padding]
	scope = []
	in_bonds = []
	all_bonds = [(-1,-1)]

	total_atoms = 0
	for smiles in mol_batch:
		#print("dkm", smiles)
		mol= get_mol_from_smiles(smiles)
		n_atoms = mol.GetNumAtoms()
		for atom in mol.GetAtoms():
			fatoms.append(atom_features(atom))
			in_bonds.append([])

		for bond in mol.GetBonds():
			a1 = bond.GetBeginAtom()
			a2 = bond.GetEndAtom()
			x = a1.GetIdx() + total_atoms
			y = a2.GetIdx() + total_atoms

			b = len(all_bonds)
			all_bonds.append((x,y))
			fbonds.append(torch.cat([fatoms[x], bond_features(bond)],0))
			in_bonds[y].append(b)

			b = len(all_bonds)
			all_bonds.append((y,x))
			fbonds.append(torch.cat([fatoms[y], bond_features(bond)],0))
			in_bonds[x].append(b)


		scope.append((total_atoms,n_atoms))
		total_atoms += n_atoms
	total_bonds = len(all_bonds)
	fatoms = torch.stack(fatoms, 0)
	fbonds = torch.stack(fbonds, 0)
	agraph = torch.zeros(total_atoms, MAX_NB).long()
	bgraph = torch.zeros(total_bonds, MAX_NB).long()

	for a in range(total_atoms):
		for i, b in enumerate(in_bonds[a]):
			agraph[a, i] = b
	for b1 in range(1, total_bonds):
		x,y = all_bonds[b1]
		for i, b2 in enumerate(in_bonds[x]):
			if all_bonds[b2][0] != y:
				bgraph[b1, i] = b2

	return fatoms, fbonds, agraph, bgraph, scope

class Discriminator(nn.Module):
	def __init__(self, latent_size, hidden_size):
		super(Discriminator, self).__init__()
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.W1 = nn.Linear(latent_size, hidden_size, bias=True)
		self.W2 = nn.Linear(hidden_size, 1, bias=True)
	def forward(self, latents):
		out = self.W1(latents)
		out = nn.ReLU()(out)
		out = self.W2(out)
		return nn.Sigmoid()(out)


class PP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(PP, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.W1 = nn.Linear(input_dim, hidden_dim, bias=True)
		self.W2 = nn.Linear(hidden_dim, output_dim, bias=True)

	def forward(self, latents):
		out = self.W1(latents)
		out = nn.ReLU()(out)
		out = self.W2(out)
		out = nn.ReLU()(out)
		return out


class MPN(nn.Module):
	def __init__(self, hidden_size, depth):
		super(MPN, self).__init__()
		self.hidden_size = hidden_size
		self.depth = depth

		self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden_size, bias=False)
		self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_DIM + hidden_size, hidden_size)

	def forward(self, mol_batch):
		fatoms, fbonds, agraph, bgraph, scope = mol2graph(mol_batch)
		fatoms = create_var(fatoms)
		fbonds = create_var(fbonds)
		agraph = create_var(agraph)
		bgraph = create_var(bgraph)

		binput = self.W_i(fbonds)
		message = nn.ReLU()(binput)

		for i in range(self.depth-1):
			# given nei_message
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = nn.ReLU()(binput + nei_message)

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = nn.ReLU()(self.W_o(ainput))
		mol_vecs = []
		for st, le in scope:
			mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0)/le
			mol_vecs.append(mol_vec)
		return torch.stack(mol_vecs, dim=0)


