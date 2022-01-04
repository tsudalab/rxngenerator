import rdkit
import rdkit.Chem as Chem
from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns,get_template_order
from collections import deque



def get_index(smiles_list, query):
	ids = []
	for i, smiles in enumerate(smiles_list):
		if smiles == query:
			ids.append(i)
	if len(ids) == 0:
		return [-1]
	return ids


class StartingReactants(object):
	def __init__(self, reactant_list):
		self.reactant_list = reactant_list
		self.reactant_list.append('unknown')
		self.vmap = {x:i for i,x in enumerate(self.reactant_list)}

	def get_index(self, smiles):
		if smiles not in self.reactant_list:
			return self.vmap['unknown']

		return self.vmap[smiles]

	def get_reactant(self, index):
		return self.reactant_list[index]

	def size(self):
		return len(self.vmap)

class Templates(object):
	def __init__(self, template_list, n_reacts):
		self.template_list = template_list
		self.vmap = {x:i for i, x in enumerate(self.template_list)}
		self.template2nreacts = {self.vmap[x]:i for x,i in zip(template_list, n_reacts)}

	def get_index(self, template):
		return self.vmap[template]
	def get_n_reacts(self, template_id):
		return self.template2nreacts[template_id]

	def get_template(self, index):
		return self.template_list[index]

	def size(self):
		return len(self.template_list)


class MoleculeNode(object):
	def __init__(self, smiles, id):
		self.smiles = smiles
		#self.mol = get_mol_from_smiles(smiles)
		self.parents = []
		self.children = []
		self.is_leaf = False
		self.type = "molecule"
		self.id = id
		self.reactant_id = -1

class TemplateNode(object):
	def __init__(self, template, id):
		self.template = template
		self.parents = []
		self.children = []
		self.type = "template"
		self.id = id
		self.depth = -1
		self.template_type = -1


class ReactionTree(object):
	def __init__(self, route):
		#print(route)
		self.route = route
		self.smiles_map = []
		self.visit = []
		# id of molecule and template nodes
		self.molecule_nodes = []
		self.template_nodes = []
		mol_count = 0
		tem_count = 0
		for rxnid, reaction in enumerate(self.route):
			product = reaction[0]
			reactants = reaction[1].split(".")
			template = reaction[2]
			ids = get_index(self.smiles_map, product)
			if ids[0] == -1:
				# add product node
				prod_node = MoleculeNode(product, mol_count)
				
				self.smiles_map.append(product)
				self.molecule_nodes.append(prod_node)
				self.visit.append(0)
				#print("adding node:", prod_node.id, prod_node.smiles)
				mol_count += 1

				temp_node = TemplateNode(template, tem_count)
				tem_count += 1
				temp_node.parents.append(prod_node)
				prod_node.children.append(temp_node)
				self.template_nodes.append(temp_node)
				for reactant in reactants:
						rec_node = MoleculeNode(reactant, mol_count)
						rec_node.parents.append(temp_node)
						temp_node.children.append(rec_node)
						self.molecule_nodes.append(rec_node)
						self.visit.append(0)
						#print("adding node:", rec_node.id, rec_node.smiles)
						self.smiles_map.append(reactant)
						mol_count += 1
			else:
				for idx in ids:
					#print(idx, self.molecule_nodes[idx].smiles, self.visit[idx])
					prod_node = self.molecule_nodes[idx]
					if self.visit[idx] == 1:
						continue
					self.visit[idx] = 1

					#print("Product:", prod_node.smiles, idx)
					# add template
					temp_node = TemplateNode(template, tem_count)
					tem_count += 1
					temp_node.parents.append(prod_node)
					prod_node.children.append(temp_node)
					self.template_nodes.append(temp_node)
					#print("Template:", temp_node.template)
					for reactant in reactants:
						rec_node = MoleculeNode(reactant, mol_count)
						rec_node.parents.append(temp_node)
						temp_node.children.append(rec_node)
						self.molecule_nodes.append(rec_node)
						self.visit.append(0)
						#print("*** adding node:", idx, rec_node.id, rec_node.smiles)
						self.smiles_map.append(reactant)
						mol_count += 1
					break

				#print("Reactant:", rec_node.smiles)
	def show_reaction(self, product, templateDic, reactDic):
		if len(product.children) == 0:
			print(product.id, reactDic.get_index(product.smiles), "leaf")
		else:
			template_node = product.children[0]
			print(template_node.id, templateDic.get_index(template_node.template), product.id, product.smiles)
			for child in template_node.children:
				print("******", product.id, child.id, child.id, len(template_node.children))
				self.show_reaction(child, templateDic, reactDic)

def extract_templates(rxns):
	templates = []
	n_reacts = []
	counts={}
	
	for rxn in rxns:
		template_nodes = rxn.template_nodes
		for template_node in template_nodes:
			if template_node.template not in templates:
				templates.append(template_node.template)
				n_reacts.append(len(template_node.children))
				counts[template_node.template] = 1
			else:
				counts[template_node.template] += 1
	
			
	return templates, n_reacts

def stats(rxns, templateDic):
	stat={}
	for rxn in rxns:
		template_nodes = rxn.template_nodes
		for template_node in template_nodes:
			idx = templateDic.get_index(template_node.template)
			if idx not in stat.keys():
				stat[idx] = 0
			else:
				stat[idx]+=1
	return stat


def extract_starting_reactants(rxns):
	starting_reactants = []
	counts ={}
	for rxn_id, rxn in enumerate(rxns):
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

	return starting_reactants




'''
route = [['CN1CCCCC1CCCc1ccccc1NC(=O)C=Cc1ccccc1', 'COc1ccc(C(=O)Nc2ccccc2CCCC2CCCCN2C)cc1.CN1CCCCC1CCCc1ccccc1N', '([#7:8]-[C:7](=[O;D1;H0:9])-[CH;D2;+0:6]=[CH;D2;+0:10]-[c;H0;D3;+0:11]1:[cH;D2;+0:2]:[cH;D2;+0:1]:[cH;D2;+0:5]:[c:4]:[cH;D2;+0:3]:1)>>C-N1-C-C-C-C-C-1-C-C-C-[c;H0;D3;+0:1]1:c:c:c:c:[c;H0;D3;+0:2]:1-N.C-O-[c;H0;D3;+0:3]1:[c:4]:[cH;D2;+0:5]:[c;H0;D3;+0:6](-[C:7](-[#7:8])=[O;D1;H0:9]):[cH;D2;+0:10]:[cH;D2;+0:11]:1'], ['COc1ccc(C(=O)Nc2ccccc2CCCC2CCCCN2C)cc1', 'COc1ccc(C(=O)Cl)cc1.CN1CCCCC1CCCc1ccccc1N', '([O;D1;H0:2]=[C;H0;D3;+0:1](-[c:3])-[NH;D2;+0:4]-[c:5])>>Cl-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3].[NH2;D1;+0:4]-[c:5]'], ['CN1CCCCC1CCCc1ccccc1N', 'CN1CCCCC1CCCBr.Nc1ccccc1Br', '([C:2]-[CH2;D2;+0:1]-[c;H0;D3;+0:3](:[c:4]):[c:5])>>Br-[CH2;D2;+0:1]-[C:2].Br-[c;H0;D3;+0:3](:[c:4]):[c:5]'], ['CN1CCCCC1CCCc1ccccc1N', 'CN1CCCCC1CCCBr.Nc1ccccc1Br', '([C:2]-[CH2;D2;+0:1]-[c;H0;D3;+0:3](:[c:4]):[c:5])>>Br-[CH2;D2;+0:1]-[C:2].Br-[c;H0;D3;+0:3](:[c:4]):[c:5]']]
rxn = ReactionTree(route)

mol_set = rxn.molecule_nodes
tem_set = rxn.template_nodes

for mol_node in mol_set:
	print(mol_node.smiles, mol_node.id)
for tem_node in tem_set:
	print(tem_node.template, tem_node.id)
'''



			





#print("test")
#routes, templates = read_multistep_rxns("synthetic_routes.txt")
#print(routes)
#RxnTree = ReactionTree(routes[0])
#print(routes[0])
#RxnTree.show_reaction(RxnTree.molecule_nodes[0])


'''
templates = {}
counts = {}
for full_rxn in routes:
	for rxn in full_rxn:
		temp = rxn[2]
		n_reactants = len(rxn[1].split("."))
		if temp not in templates:
			templates[temp] = len(templates)
			counts[temp] = [n_reactants]
		else:
			tmp = counts[temp]
			tmp.append(n_reactants)
			counts[temp] = tmp


print(counts)
'''