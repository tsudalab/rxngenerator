import rdkit
import rdkit.Chem as Chem
from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates
from collections import deque
import torch
import torch.nn as nn
from nnutils import create_var
MAX_REACTANTS = 7


class RXNEncoder(nn.Module):
	def __init__(self, hidden_size, latent_size,reactantDic, templateDic, r_embedding = None, t_embedding=None):
		super(RXNEncoder, self).__init__()
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.reactantDic = reactantDic
		self.templateDic = templateDic
		self.has_mpn = False
		#self.r_embedding = r_embedding
		if r_embedding is None:
			self.r_embedding = nn.Embedding(reactantDic.size(), hidden_size)
			self.has_mpn = False
		else:
			self.r_embedding = r_embedding
			self.has_mpn=True
		if t_embedding is None:
			self.t_embedding = nn.Embedding(templateDic.size(), hidden_size)
		else:
			self.t_embedding = t_embedding

		self.W_m = nn.Linear(self.hidden_size, self.hidden_size)
		self.W_t = nn.Linear(self.hidden_size, self.hidden_size)
		self.W_l = nn. Linear(self.hidden_size, self.latent_size)



	def forward(self, rxn_tree_batch):
		#print("**********************************start")
		orders = []
		for rxn_tree in rxn_tree_batch:
			order = get_template_order(rxn_tree)
			orders.append(order)
		max_depth = max([len(order) for order in orders])
		h = {}
		padding = create_var(torch.zeros(self.hidden_size), False)
		for t in range(max_depth-1,-1, -1):
			template_ids = []
			rxn_ids = []
			for i, order in enumerate(orders):
				if t < len(order):
					template_ids.extend(order[t])
					rxn_ids.extend([i]*len(order[t]))
			cur_mols = []
			cur_tems = []
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				#print(t, rxn_id, template_id)
				template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
				cur_mol = []
				for reactant in template_node.children:
					if len(reactant.children) == 0: # leaf node
						#print(reactant.smiles)
						#print(self.reactantDic.reactant_list)
						if self.has_mpn == False:
							reactant_id = self.reactantDic.get_index(reactant.smiles)
							mfeature = self.r_embedding(create_var(torch.LongTensor([reactant_id])))[0]
						else:
							mfeature = self.r_embedding([reactant.smiles])[0]
						h[(rxn_id, reactant.id)] = mfeature
					cur_mol.append(h[(rxn_id, reactant.id)])
				pad_length = MAX_REACTANTS - len(cur_mol)
				cur_mol.extend([padding]*pad_length)
				temp_id = self.templateDic.get_index(template_node.template)
				tfeat = self.t_embedding(create_var(torch.LongTensor([temp_id])))[0]
				cur_mols.extend(cur_mol)
				cur_tems.append(tfeat)
				#print(cur_mols, template_id, rxn_id)

			#print(t, cur_mols)
			#exit(1)

			#cur_mols1 = torch.cat(cur_mols, dim=0).view(-1,self.hidden_size)
			#cur_tems1 = torch.cat(cur_tems, dim=0).view(-1, self.hidden_size)

			cur_mols = torch.stack(cur_mols, dim = 0)
			cur_tems = torch.stack(cur_tems, dim=0)
			#cur_mols = cur_mols.view(-1, MAX_REACTANTS * self.hidden_size)

			o_tems = self.W_t(cur_tems)
			o_mols = self.W_m(cur_mols)

			o_mols = o_mols.view(-1, MAX_REACTANTS, self.hidden_size)
			o_mols= o_mols.sum(dim=1)
			new_h = nn.ReLU()(o_tems + o_mols)
			i = 0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
				product = template_node.parents[0]
				h[(rxn_id, product.id)] = new_h[i]
				i+=1
		mol_vecs = []
		for i in range(len(rxn_tree_batch)):
			mol_vecs.append(h[(i, 0)])
		mol_vecs = torch.stack(mol_vecs, dim=0)
		#latent_z = self.W_l(mol_vecs)
		#print("**********************************end")
		return mol_vecs







'''

print("test")
routes, templates = read_multistep_rxns("synthetic_routes.txt")
rxn_tree_batch = [ReactionTree(route) for route in routes[:10]]
reactants = extract_starting_reactants(rxn_tree_batch)

reactantDic = StartingReactants(reactants)
templateDic = Templates(templates)

hidden_size = 10
n_reactants = len(reactants)
n_templates = len(templates)

print(len(reactants), len(templates))

encoder = Encoder( hidden_size, reactantDic, templateDic)
encoder.forward(rxn_tree_batch)

'''












