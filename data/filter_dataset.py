import sys
sys.path.append('../rxnft_vae')
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
import networkx as nx
from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, check
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
import numpy as np
from collections import deque
from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed


def print_reaction(rxn):
	# get order
	mol_nodes = rxn.molecule_nodes
	tem_nodes = rxn.template_nodes

	order ={}
	root = tem_nodes[0]
	queue = deque([root])
	visited = set([root.id])

	order[0] =[root.id]
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
	
	reactions_str = []
	for t in range(len(order)):
		for template_id in order[t]:
			
			template_node = tem_nodes[template_id]
			template_str = template_node.template
			reactant_str =[]
			for reactant_node in template_node.children:
				reactant_str.append(reactant_node.smiles)
			reactant_str = ".".join(reactant_str)
			product_str = template_node.parents[0].smiles

			reaction_str= "$".join([product_str, reactant_str, template_str])
			reactions_str.append(reaction_str)

	return " ".join(reactions_str)

def filter_dataset(input_files, output_file, metric="qed", reactant_count = 5, template_count = 5):# for uspto
	print("input files:", input_files)
	print("output file", output_file)

	routes =[]
	template_counts={}
	count = 0
	for input_file in input_files:
		with open(input_file, "r") as reader:
			lines = reader.readlines()

			for line in lines:
	
				reactions = line.strip().split(' ')
				full_rxn = []
				valid = True
				for reaction in reactions:
					abc = reaction.split("*")
					product = abc[0]
					reactants = abc[1]
					template = abc[2]
					if template not in template_counts.keys():
						template_counts[template] = 1
					else:
						template_counts[template] +=1
					full_rxn.append([product, reactants, template])
					n1 = len(reactants.split("."))
					p1,p2 = template.split(">>")
					n2 = len(p2.split("."))
					if n1 !=n2:
						valid = False
				#print(full_rxn, len(full_rxn))
				if valid:
					routes.append(full_rxn)
	rxn_trees_t=[]
	for route in routes:
		tree = ReactionTree(route)
		rxn_trees_t.append(tree)
	#rxn_trees_t = [ReactionTree(route) for route in routes]
	rxn_trees =[]
	count = 0
	rxn_ids=[]
	for i, rxn_tree in enumerate(rxn_trees_t):
		smiles = rxn_tree.molecule_nodes[0].smiles
		if can_be_decomposed(smiles):
			rxn_trees.append(rxn_tree)
			rxn_ids.append(i)
			count +=1
			print(smiles, count)
	#import pickle
	#with open('valid_rxn_ids_uspto.dat', 'wb') as f:
	#	rxn_ids=pickle.load(f)
	#	pickle.dump(rxn_ids, f)


	print("total rxns:", len(rxn_trees))

	###########################
	starting_reactants = []
	counts ={}
	for rxn_id, rxn in enumerate(rxn_trees):
		mol_nodes = rxn.molecule_nodes
		root = mol_nodes[0]
		queue = deque([root])
		while len(queue) > 0:
			x = queue.popleft()
				#exit(1)
			if len(x.children) == 0:
				smiles = x.smiles
				if smiles not in starting_reactants:
					starting_reactants.append(smiles)
					counts[smiles] =1
				else:
					counts[smiles] +=1
			else:
				template = x.children[0]
				for y in template.children:
					queue.append(y)
					#visisted.add(y.id
	
	new_rxn_trees=[]
	for rxn_id, rxn in enumerate(rxn_trees):
		mol_nodes = rxn.molecule_nodes
		template_nodes = rxn.template_nodes
		#if len(template_nodes) < 2:
		#	continue
		root = mol_nodes[0]
		queue = deque([root])
		flag = True
		while len(queue) > 0:
			x = queue.popleft()
				#exit(1)
			if len(x.children) == 0:
				smiles = x.smiles
				if counts[smiles] < reactant_count:
					flag = False
					break
			else:
				template = x.children[0]
				for y in template.children:
					queue.append(y)
		if flag:
			new_rxn_trees.append(rxn)
	print("After filtering out dataset based on reactants",len(new_rxn_trees), len(rxn_trees))
	
	########################
	templates = []
	n_reacts = []
	counts={}
	
	for rxn in new_rxn_trees:
		template_nodes = rxn.template_nodes
		for template_node in template_nodes:
			if template_node.template not in templates:
				templates.append(template_node.template)
				n_reacts.append(len(template_node.children))
				counts[template_node.template] = 1
			else:
				counts[template_node.template] += 1

	new_new_rxn_trees=[]

	for rxn in new_rxn_trees:
		flag = True
		template_nodes = rxn.template_nodes
		for template_node in template_nodes:
			if counts[template_node.template] < template_count:
				flag=False
				break
		if flag:
			new_new_rxn_trees.append(rxn)

	print("After filtering out dataset based on templates",len(new_new_rxn_trees))

			

	###########################


	smiles_list =[]
	scores =[]
	if metric =="logp":
		logP_values = np.loadtxt('logP_values.txt')
		SA_scores = np.loadtxt('SA_scores.txt')
		cycle_scores = np.loadtxt('cycle_scores.txt')
		logp_m = np.mean(logP_values)
		logp_s = np.std(logP_values)
		sascore_m = np.mean(SA_scores)
		sascore_s = np.std(SA_scores)
		cycle_m = np.mean(cycle_scores)
		cycle_s = np.std(cycle_scores)

	for rxn in new_new_rxn_trees:
		smiles = rxn.molecule_nodes[0].smiles
		smiles_list.append(smiles)
		if metric=="logp":
			scores.append(get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s))
		if metric =="qed": 
			qed =QED.qed(rdkit.Chem.MolFromSmiles(smiles))
			scores.append(qed)

	count = 0
	with open(output_file, "w") as writer:
		for rxn, score in zip(new_new_rxn_trees, scores):
			if check(rxn):
				count+=1
				reactions_str = print_reaction(rxn)
				reactions_str = " ".join([reactions_str, str(score)])
				writer.write(reactions_str)
				writer.write("\n")
	writer.close()
	print("Number of original chemical reactions:", len(rxn_trees))
	print("Number of valid chemical reactions:", count)

filter_dataset(["synthetic_routes.txt"], "data.txt", metric ="qed")







