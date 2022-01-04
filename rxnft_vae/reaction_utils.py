import numpy as np
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from collections import deque
import itertools
from rdkit.Chem import rdChemReactions
import networkx as nx
from rdkit.Chem import QED


def get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s):
	logp_value = Descriptors.MolLogP(MolFromSmiles(smiles))
	sascore = -sascorer.calculateScore(MolFromSmiles(smiles)) 
	cycle_score = 0


	cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles))))
	if len(cycle_list)==0:
		cycle_length = 0
	else:
		cycle_length = max([len(j) for j in cycle_list])
		if cycle_length <= 6:
			cycle_length = 0
		else:
			cycle_length = cycle_length - 6
	cycle_score = -cycle_length

	logP_value_normalized = (logp_value - logp_m)/logp_s
	sascore_normalized = (sascore - sascore_m)/sascore_s
	cycle_score_normalized = (cycle_score- cycle_m)/cycle_s

	return logP_value_normalized + sascore_normalized + cycle_score_normalized


def get_qed_score(smiles):
	score = QED.qed(rdkit.Chem.MolFromSmiles(smiles))
	return score


def score(smiles_list):
	logP_values = [Descriptors.MolLogP(MolFromSmiles(smiles)) for smiles in smiles_list]
	sascores = [-sascorer.calculateScore(MolFromSmiles(smiles)) for smiles in smiles_list]
	cycle_scores = []


	for smiles in smiles_list:
		cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles))))
		if len(cycle_list)==0:
			cycle_length = 0
		else:
			cycle_length = max([len(j) for j in cycle_list])
			if cycle_length <= 6:
				cycle_length = 0
			else:
				cycle_length = cycle_length - 6

		cycle_scores.append(-cycle_length)
	data = [np.mean(logP_values),np.std(logP_values),np.mean(sascores),np.std(sascores),np.mean(cycle_scores),np.std(cycle_scores)]
	data = np.array(data)

	with open("mean_std.npy", "wb") as f:
		np.save(f, data)

	logP_values_normalized = (np.array(logP_values)-np.mean(logP_values))/np.std(logP_values)
	sascores_normalized = (np.array(sascores) - np.mean(sascores))/np.std(sascores)
	cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores))/np.std(cycle_scores)

	return logP_values_normalized, sascores_normalized, cycle_scores_normalized
def check(rxn):
	molecule_nodes = rxn.molecule_nodes
	template_nodes = rxn.template_nodes
	root = template_nodes[0]
	queue = deque([root])
	root.depth = 0
	order ={}
	visited = set([root.id])
	node2smiles={}
	order[0]=[root.id]

	while len(queue) > 0:
		x = queue.popleft()
		for y in x.children:
			if len(y.children) == 0:
				continue
			template = y.children[0]
			if template.id not in visited:
				queue.append(template)
				visited.add(template.id)
				template.depth = x.depth + 1
				if template.depth not in order:
					order[template.depth] = [template.id]
				else:
					order[template.depth].append(template.id)
	#print(order)
	maxdepth = len(order) - 1
	for t in range(maxdepth, -1, -1):
		for template_id in order[t]:
			template_node = template_nodes[template_id]
			template = template_node.template
			
			reactants =[]
			for reactant_node in template_node.children:
				if len(reactant_node.children) == 0:
					reactant = reactant_node.smiles
					reactants.append(reactant)
					node2smiles[reactant_node.id] = reactant
					#print(reactant)

				else:
					reactant = node2smiles[reactant_node.id]
					reactants.append(reactant)
			possible_templates = reverse_template(template)
			reacts = [Chem.MolFromSmiles(reactant) for reactant in reactants]
			#print(reactants)
			#print(template)
			product = None
			for template_ in possible_templates:
				try:
					rn = rdChemReactions.ReactionFromSmarts(template_)
					products = rn.RunReactants(reacts)
					if len(products) > 0:
						product = products[0][0]
						break
				except:
					return False
			if product == None:
				return False
			else:
				product_id = template_node.parents[0].id
				node2smiles[product_id] = Chem.MolToSmiles(product)
	#print(node2smiles)
	print(node2smiles[0], molecule_nodes[0].smiles)


	return True





def filter_dataset(rxn_trees):
	return_rxns=[]
	for i, rxn in enumerate(rxn_trees):
		if check(rxn):
			#print(i, "OK")
		#else:
		#	print(i, "not OK")
			return_rxns.append(rxn)
	return return_rxns

def get_mol_from_smiles(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	Chem.Kekulize(mol)
	return mol

def get_smiles_from_mol(mol):
	return Chem.MolToSmiles(mol, kekuleSmiles=True)



def read_multistep_rxns(filename):
	synthetic_routes = []
	scores = []
	with open(filename, 'r') as reader:
		lines = reader.readlines()
		for line in lines:

			full_rxn = []
			reactions = line.strip().split(' ')
			for reaction in reactions[:-1]:
				product, reactants, template = reaction.split('*')
				full_rxn.append([product, reactants, template])
			synthetic_routes.append(full_rxn)
			scores.append(float(reactions[-1]))
	return synthetic_routes, scores

def reverse_template(template):
	p1, p2 = template.split(">>")
	p2 = p2.split(".")
	p2_list =  list(itertools.permutations(p2))
	reactant_list = [".".join(p2) for p2 in p2_list]

	return [">>".join([p2, p1])  for p2 in reactant_list]

def get_possible_reactants(reactants):
	possible=[]
	n = len(reactants)
	order ={}
	it={}
	for i in range(n):
		order[i] = reactants[i]
		it[i] = 0
	




def get_template_order(rxn):
	mol_nodes = rxn.molecule_nodes
	tem_nodes = rxn.template_nodes
	#for template_node in tem_nodes:
	#	print(template_node.id)

	order={}
	root = tem_nodes[0]
	queue = deque([root])
	visisted = set([root.id])
	root.depth = 0
	order[0] =[root.id]
	
	while len(queue) > 0:
		x = queue.popleft()
		#print("pop:", x.id)
		for y in x.children:
			if len(y.children) == 0: # starting molecule
				continue
			template = y.children[0] 
			if template.id not in visisted:
				queue.append(template)
				#print("push:", template.id)
				visisted.add(template.id)
				template.depth = x.depth + 1
				if template.depth not in order:
					order[template.depth] = [template.id]
				else:
					order[template.depth].append(template.id)

	

	#print(order)
	#exit(1)
	return order







