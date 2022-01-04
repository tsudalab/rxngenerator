import rdkit
import rdkit.Chem as Chem
import copy
from chemutils import get_mol, decode_stereo, tree_decomp, get_clique_mol, get_smiles, set_atommap, enum_assemble




def get_slots(smiles):
	mol = Chem.MolFromSmiles(smiles)
	#print(smiles, mol)
	results = [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]
	return results


# definition for FragmentVocab
class FragmentVocab(object):
	def __init__(self, smiles_list, filename=None):
		if filename is not None:
			smiles_list = self.load(filename)

		self.vocab = smiles_list
		self.vmap = {x:i for i,x in enumerate(self.vocab)}
		self.slots = [get_slots(smiles) for smiles in self.vocab]
	def get_index(self, smiles):
		return self.vmap[smiles]
	def get_smiles(self, idx):
		return self.vocab[idx]
	def size(self):
		return len(self.vocab)
	def save(self, filename):
		with open(filename, "w") as f:
			for w in self.vocab:
				f.write(w + "\n")

	def load(self, filename):
		smiles_list =[]
		with open(filename,"r") as f:
			lines = f.readlines()
			for line in lines:
				smiles_list.append(line.strip())
		return smiles_list
	def get_slots(self, idx):
		return copy.deepcopy(self.slots[idx])



class FragmentNode(object):
	def __init__(self, smiles, clique=[]):
		self.smiles = smiles
		self.clique = [x for x in clique]
		self.mol = get_mol(smiles)
		self.neighbors =[]
		self.wid = -1
		self.idx = -1
	def add_neighbor(self, nei_node):
		self.neighbors.append(nei_node)
	def print(self):
		print("smiles:", self.smiles)
		print("wid:", self.wid)
		print("idx:", self.idx)
		print("connected to:")
		for i in range(len(self.neighbors)):
			print(self.neighbors[i].idx, self.neighbors[i].wid, self.neighbors[i].smiles)
	def assemble(self):
		neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
		neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
		singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
		neighbors = singletons + neighbors
		print("------neighbors:")
		for neighbor in neighbors:
			neighbor.print()
			print("---------")
		cands,aroma = enum_assemble(self, neighbors)
		new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
		if len(new_cands) > 0: cands = new_cands
		if len(cands) > 0:
			self.cands, _ = zip(*cands)
			self.cands = list(self.cands)
		else:
			self.cands = []
		print("candiatae:",self.cands)





class FragmentTree(object):
	def __init__(self, smiles):
		if smiles is None:
			self.nodes = []
		else:
			self.smiles = smiles
			self.mol = get_mol(smiles)


			mol = Chem.MolFromSmiles(smiles)
			self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles = True)
			self.smiles2D = Chem.MolToSmiles(mol)
			self.stereo_cands = decode_stereo(self.smiles2D)


			cliques, edges = tree_decomp(self.mol)
			self.nodes =[]
			root = 0

			for i,c in enumerate(cliques):
				cmol = get_clique_mol(self.mol, c)
				csmiles = get_smiles(cmol)
				node = FragmentNode(csmiles, c)
				self.nodes.append(node)
				if min(c) == 0:
					root = i
			for x, y in edges:
				self.nodes[x].add_neighbor(self.nodes[y])
				self.nodes[y].add_neighbor(self.nodes[x])
			if root > 0:
				self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]
			for i, node in enumerate(self.nodes):
				node.nid = i+1
				if len(node.neighbors) > 1:
					set_atommap(node.mol, node.nid)
				node.is_leaf = (len(node.neighbors)==1)
	def print(self):
		for node in self.nodes:
			for nei in node.neighbors:
				print(node.idx, node.smiles, "-->", nei.idx, nei.smiles)
	def assemble(self):
		for node in self.nodes:
			node.print()
			node.assemble()
			print("Done")



def can_be_decomposed(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return False
	#smiles3D = Chem.MolToSmiles(mol, isomericSmiles = True)
	#smiles2D = Chem.MolToSmiles(mol)
	#stereo_cands = decode_stereo(smiles2D)
	cliques, edges = tree_decomp(mol)
	#print(cliques, edges)
	for i,c in enumerate(cliques):
		#print(smiles, i, c)
		cmol = get_clique_mol(mol, c)
		if cmol is None:
			return False
		csmiles = get_smiles(cmol)
		cmol = Chem.MolFromSmiles(csmiles)
		if cmol is None:
			return False
	return True

#smiles = "CN1CCC(CCCN(C(=O)OC(C)(C)C)C(=NC(=O)OC(C)(C)C)NC(=O)OC(C)(C)C)CC1"
#smiles = "COc1cccc(C(=O)OC[C@@H]2CCCN(C(=O)c3ccc(C)o3)C2)n1"
#result = can_be_decomposed(smiles)
#tree = FragmentTree(smiles)
#tree.print()
#tree.assemble()

'''
cset=set()
with open("train.txt", "r") as reader:
	lines = reader.readlines()
	for i, line in enumerate(lines):
		print(line)
		ftree = FragmentTree(line.strip())
		print(ftree.smiles)
		for c in ftree.nodes:
			print(c.smiles)
'''
