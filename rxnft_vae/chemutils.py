import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000


def set_atommap(mol, num = 0):
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(num)


def get_mol(smiles):
	mol = Chem.MolFromSmiles(smiles)
	#print("ok2", mol)
	if mol is None:
		return None
	Chem.Kekulize(mol)
	return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)
    #return Chem.MolToSmiles(mol)


def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
    	a1 = bond.GetBeginAtom().GetIdx()
    	a2 = bond.GetEndAtom().GetIdx()
    	bt = bond.GetBondType()
    	new_mol.AddBond(a1, a2, bt)
    	#if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
    	#	bt = Chem.rdchem.BondType.SINGLE
    return new_mol
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])
def get_clique_mol(mol, atoms):
	smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
	new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
	new_mol = copy_edit_mol(new_mol).GetMol()
	#print(smiles, new_mol,"before")
	new_mol = sanitize(new_mol)
	#print(smiles, new_mol, "after")
	return new_mol

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D



def tree_decomp(mol):
	#print("decomposing tree")
	n_atoms = mol.GetNumAtoms()
	
	if n_atoms==1:
		return [[0]],[]
	cliques = []
	for bond in mol.GetBonds():
		a1 = bond.GetBeginAtom().GetIdx()
		a2 = bond.GetEndAtom().GetIdx()
		if not bond.IsInRing():
			cliques.append([a1, a2])
	ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
	cliques.extend(ssr)


	nei_list = [[] for i in range(n_atoms)]
	for i in range(len(cliques)):
		for atom in cliques[i]:
			nei_list[atom].append(i)
	# merging rings with intersection > 2 atoms
	for i in range(len(cliques)):
		if len(cliques[i]) <= 2: continue
		for atom in cliques[i]:
			for j in nei_list[atom]:
				if j <=i or len(cliques[j])<=2:continue
				inter = set(cliques[i]) & set(cliques[j])
				if len(inter) > 2:
					cliques[i].extend(cliques[j])
					cliques[i] = list(set(cliques[i]))
					cliques[j] =[]
	cliques = [c for c in cliques if len(c) > 0]
	nei_list =[[] for i in range(n_atoms)]
	for i in range(len(cliques)):
		for atom in cliques[i]:
			nei_list[atom].append(i)

	# building edges and add singleton cliques
	edges = defaultdict(int)
	for atom in range(n_atoms):
		if len(nei_list[atom]) <=1:
			continue
		cnei = nei_list[atom]
		bonds = [c for c in cnei if len(cliques[c])==2]
		rings = [c for c in cnei if len(cliques[c]) > 4]
		if len(bonds) > 2 or (len(bonds)==2 and len(cnei) > 2):
			cliques.append([atom])
			c2 = len(cliques) - 1
			for c1 in cnei:
				edges[(c1,c2)] = 1
		elif len(rings) > 2: # multiple complex rings
			cliques.append([atom])
			c2 = len(cliques) - 1
			for c1 in cnei:
				edges[(c1,c2)] = MST_MAX_WEIGHT - 1
		else:
			for i in range(len(cnei)):
				for j in range(i+1, len(cnei)):
					c1,c2 = cnei[i], cnei[j]
					inter = set(cliques[c1]) & set(cliques[c2])
					if edges[(c1,c2)] < len(inter):
						edges[(c1,c2)] = len(inter)

	edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
	if len(edges) == 0:
		return cliques, edges
	# compute MST
	row, col, data = zip(*edges)
	n_clique = len(cliques)
	clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
	junc_tree = minimum_spanning_tree(clique_graph)
	row,col = junc_tree.nonzero()
	edges = [(row[i],col[i]) for i in range(len(row))]
	return (cliques, edges)
def check_aroma(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() >= 3]
    if len(rings) < 2: return 0 #Only multi-ring system needs to be checked

    get_nid = lambda x: 0 if x.is_leaf else x.nid
    benzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.benzynes] 
    penzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.penzynes] 
    if len(benzynes) + len(penzynes) == 0: 
        return 0 #No specific aromatic rings

    n_aroma_atoms = 0
    for atom in cand_mol.GetAtoms():
        if atom.GetAtomMapNum() in benzynes+penzynes and atom.GetIsAromatic():
            n_aroma_atoms += 1

    if n_aroma_atoms >= len(benzynes) * 4 + len(penzynes) * 3:
        return 1000
    else:
        return -0.001 
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
	all_attach_confs = []
	singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]


	def search(cur_amap, depth):
		if len(all_attach_confs) > MAX_NCAND:
			return
		if depth == len(neighbors):
			all_attach_confs.append(cur_amap)
			return
		nei_node = neighbors[depth]
		cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
		cand_smiles = set()
		candidates = []
		#print("num cand_amap:", len(cand_amap))
		for i, amap in enumerate(cand_amap):
			cand_mol = local_attach(node.mol, neighbors[:depth+1], prev_nodes, amap)
			cand_mol = sanitize(cand_mol)
			if cand_mol is None:
				#print("dkm toan None")
				continue
			smiles = get_smiles(cand_mol)
			if smiles in cand_smiles:
				continue
			cand_smiles.add(smiles)
			candidates.append(amap)
		if len(candidates) == 0:
			return
		for new_amap in candidates:
			search(new_amap, depth + 1)
	#print("dkm1 search")
	search(prev_amap, 0)
	cand_smiles = set()
	candidates = []
	for amap in all_attach_confs:

		cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
		cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
		smiles = Chem.MolToSmiles(cand_mol)
		if smiles in cand_smiles:
			continue
		cand_smiles.add(smiles)
		Chem.Kekulize(cand_mol)
		candidates.append( (smiles,cand_mol,amap) )
	#print("end enum_assemble")
	return candidates
#This version records idx mapping between ctr_mol and nei_mol
def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
	#print("atach mols")
	prev_nids = [node.nid for node in prev_nodes]
	for nei_node in prev_nodes + neighbors:
		nei_id,nei_mol = nei_node.nid,nei_node.mol
		amap = nei_amap[nei_id]
		for atom in nei_mol.GetAtoms():
			if atom.GetIdx() not in amap:
				new_atom = copy_atom(atom)
				amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)
		if nei_mol.GetNumBonds() == 0:
			nei_atom = nei_mol.GetAtomWithIdx(0)
			ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
			ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
		else:
			for bond in nei_mol.GetBonds():
				a1 = amap[bond.GetBeginAtom().GetIdx()]
				a2 = amap[bond.GetEndAtom().GetIdx()]
				if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
					ctr_mol.AddBond(a1, a2, bond.GetBondType())
				elif nei_id in prev_nids: #father node overrides
					ctr_mol.RemoveBond(a1, a2)
					ctr_mol.AddBond(a1, a2, bond.GetBondType())
	#print("end attach mols")
	return ctr_mol
def check_singleton(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() > 2]
    singletons = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() == 1]
    if len(singletons) > 0 or len(rings) == 0: return True

    n_leaf2_atoms = 0
    for atom in cand_mol.GetAtoms():
        nei_leaf_atoms = [a for a in atom.GetNeighbors() if not a.IsInRing()] #a.GetDegree() == 1]
        if len(nei_leaf_atoms) > 1: 
            n_leaf2_atoms += 1

    return n_leaf2_atoms == 0
def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
	#print("begin local attach")
	ctr_mol = copy_edit_mol(ctr_mol)
	nei_amap = {nei.nid:{} for nei in prev_nodes + neighbors}
	for nei_id,ctr_atom,nei_atom in amap_list:
		nei_amap[nei_id][nei_atom] = ctr_atom
	ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
	#print("end local attach")
	return ctr_mol.GetMol()
def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()
def enum_attach(ctr_mol, nei_node, amap, singletons):
	#print("begin enum_attach")
	nei_mol,nei_idx = nei_node.mol,nei_node.nid
	att_confs = []
	black_list = [atom_idx for nei_id,atom_idx,_ in amap if nei_id in singletons]
	ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]
	ctr_bonds = [bond for bond in ctr_mol.GetBonds()]
	if nei_mol.GetNumBonds() == 0: #neighbor singleton
		nei_atom = nei_mol.GetAtomWithIdx(0)
		used_list = [atom_idx for _,atom_idx,_ in amap]
		for atom in ctr_atoms:
			if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
				new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
				att_confs.append( new_amap )
	elif nei_mol.GetNumBonds() == 1: #neighbor is a bond
		bond = nei_mol.GetBondWithIdx(0)
		bond_val = int(bond.GetBondTypeAsDouble())
		b1,b2 = bond.GetBeginAtom(), bond.GetEndAtom()
		for atom in ctr_atoms: 
			#Optimize if atom is carbon (other atoms may change valence)
			if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
				continue
			if atom_equal(atom, b1):
				new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
				att_confs.append( new_amap )
			elif atom_equal(atom, b2):
				new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
				att_confs.append( new_amap )
	else: 
		for a1 in ctr_atoms:
			for a2 in nei_mol.GetAtoms():
				if atom_equal(a1, a2):
					#Optimize if atom is carbon (other atoms may change valence)
					if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
						continue
					new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
					att_confs.append( new_amap )
		if ctr_mol.GetNumBonds() > 1:
			for b1 in ctr_bonds:
				for b2 in nei_mol.GetBonds():
					if ring_bond_equal(b1, b2):
						new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()), (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
						att_confs.append( new_amap )
					if ring_bond_equal(b1, b2, reverse=True):
						new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()), (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
						att_confs.append( new_amap )
		#print("end enum_attach")
	return att_confs