import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import torch
import torch.nn as nn
from nnutils import create_var, attention
import math
import torch.nn.functional as F
from rdkit.Chem import rdChemReactions
from collections import deque
from mpn import MPN
from reaction import MoleculeNode, TemplateNode
from reaction_utils import *
from rdkit.Chem import AllChem

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
	return order




class RXNDecoder1(nn.Module):
	def __init__(self, hidden_size, latent_size, reactantDic, templateDic, r_embedding=None, t_embedding=None):
		super(RXNDecoder1, self).__init__()
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.reactantDic = reactantDic
		self.templateDic = templateDic
		if r_embedding is None:
			self.r_embedding = nn.Embedding(self.reactantDic.size(), hidden_size)
		else:
			self.r_embedding = r_embedding
		if t_embedding is None:
			self.t_embedding = nn.Embedding(self.templateDic.size(), hidden_size)
		else:
			self.t_embedding = t_embedding


		self.W_root = nn.Linear(self.latent_size, self.hidden_size)
		self.W_template = nn.Linear(self.latent_size+self.hidden_size*2, self.templateDic.size())
		self.W_reactant = nn.Linear(self.latent_size + 4 * self.hidden_size, self.hidden_size)
		self.W_starting_react = nn.Linear(self.latent_size + 3 * self.hidden_size, self.reactantDic.size())
		self.W_is_leaf = nn.Linear(self.latent_size + 3 * self.hidden_size, 1)


		#self.root_loss = nn.MSELoss(size_average=False)
		self.molecule_loss = nn.MSELoss(size_average=False)
		self.is_leaf_loss = nn.BCEWithLogitsLoss(size_average=False)
		self.starting_react_loss = nn.CrossEntropyLoss(size_average=False)
		self.template_loss = nn.CrossEntropyLoss(size_average=False)
		self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

	def forward(self, rxn_tree_batch, latent_vecs, encoder_outputs):
		orders = []
		B = len(rxn_tree_batch)
		for rxn_tree in rxn_tree_batch:
			order = get_template_order(rxn_tree)
			orders.append(order)
		max_depth = max([len(order) for order in orders])

		target_molecules = []
		target_templates =[]

		is_leaf ={}
		leaf_nodes ={}

		for rxn_id in range(len(rxn_tree_batch)):
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			template_nodes = rxn_tree_batch[rxn_id].template_nodes
			for ind, molecule_node in enumerate(molecule_nodes):
				target_molecules.append(molecule_node.smiles)

				if len(molecule_node.children) == 0:
					is_leaf[(rxn_id, ind)] = 1
					leaf_nodes[(rxn_id, ind)] = self.reactantDic.get_index(molecule_node.smiles)
				else:

					is_leaf[(rxn_id, ind)] = 0
					if molecule_node.id ==0:
						continue
					leaf_nodes[(rxn_id, ind)] = self.reactantDic.get_index("unknown")
			for template_node in template_nodes:
				target_templates.append(self.templateDic.get_index(template_node.template))
		target_mol_embeddings = self.r_embedding(target_molecules)
		i=0
		h_target ={}
		l_target ={}
		for rxn_id in range(len(rxn_tree_batch)):
			for j in range(len(rxn_tree_batch[rxn_id].molecule_nodes)):
				h_target[(rxn_id, j)] = target_mol_embeddings[i]
				i+=1
		i=0
		for rxn_id in range(len(rxn_tree_batch)):
			for j in range(len(rxn_tree_batch[rxn_id].template_nodes)):
				#print(rxn_id, j)
				l_target[(rxn_id, j)] = target_templates[i]
				i+=1

		# predicting roots embedding
		#padding = create_var(torch.zeros((B, self.hidden_size)))
		#r_in = torch.cat((latent_vecs, padding), dim=1)
		root_embeddings = self.W_root(latent_vecs)
		root_embeddings = nn.ReLU()(root_embeddings)


		h_pred={}# vectors for molecule nodes
		l_pred={}# vectors for template nodes
		logits_pred={}
		
		for rxn_id in range(B):
			h_pred[(rxn_id, 0)] = root_embeddings[rxn_id]


		debug=[]
		debug_tempaltes = []

		
		template_loss = 0.0



		for t in range(max_depth):
			template_targets =[]
			template_ids = []
			rxn_ids = []
			template2reactants = {}
			for i, order in enumerate(orders):
				if t < len(order):
					template_ids.extend(order[t])
					rxn_ids.extend([i]*len(order[t]))
			#print(t, rxn_ids, template_ids)

			# template prediction
			cur_hiddens = []
			cur_latents =[]
			n_reactants = []
			cur_enc_outputs_t =[]


			for template_id, rxn_id in zip(template_ids, rxn_ids):
				cur_enc_outputs_t.append(encoder_outputs[rxn_id])
				template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
				product_id = template_node.parents[0].id
				cur_hiddens.append(h_pred[(rxn_id, product_id)])
				cur_latents.append(latent_vecs[rxn_id])
				template_targets.append(self.templateDic.get_index(template_node.template))
				template2reactants[(rxn_id, template_id)] = []
				for child in template_node.children:
					template2reactants[(rxn_id, template_id)].append(child.id)
				n_reactants.append(len(template_node.children))

			cur_hiddens = torch.stack(cur_hiddens, dim=0)
			cur_latents = torch.stack(cur_latents, dim = 0)
			context = attention(cur_enc_outputs_t, cur_hiddens)
			input = torch.cat([cur_latents, cur_hiddens, context], dim=1)
			logits = self.W_template(input)
			

			#template_targets = create_var(torch.LongTensor(template_targets))
			#template_loss += self.template_loss(logits, template_targets)
			i=0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				logits_pred[(rxn_id, template_id)] = logits[i]
				i+=1
			
			


			# gumbel sampling
			output = F.gumbel_softmax(logits, tau=0.5, hard=True, eps=1e-10, dim=-1)
			_, ind = output.max(dim=-1)
			template_vecs = create_var(torch.LongTensor(ind.cpu()))


			template_vecs = self.t_embedding(template_vecs)
			i = 0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				#l[(rxn_id, template_id)] = template_vecs[i]
				l_pred[(rxn_id, template_id)] = template_vecs[i]
				i+=1
				#debug_tempaltes.append((rxn_id, template_id))


			# predict reactants
			#input = torch.cat([input, template_vecs], dim=1)
			max_n_reactants = max(n_reactants)
			for n in range(max_n_reactants):
				rxn_tem_mols= []
				prev_vectors_t = []
				latent_vectors_t = []
				product_vectors_t = []
				template_vectors_t = []
				cur_enc_outputs_t=[]
				
				#print("FINISH 1", n)
				
				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						mol_id = template_node.children[n].id
						product_id = template_node.parents[0].id
						rxn_tem_mols.append((rxn_id, template_id, mol_id))

						latent_vectors_t.append(latent_vecs[rxn_id])
						product_vectors_t.append(h_pred[(rxn_id, product_id)])
						cur_enc_outputs_t.append(encoder_outputs[rxn_id])
						#template_vectors_t.append(l_pred[(rxn_id, template_id)])
						#print(l_pred[(rxn_id, template_id)])
						#print(l_target[(rxn_id, template_id)])
						template_vectors_t.append(l_target[(rxn_id, template_id)])
						if n == 0:
							prev_vectors_t.append(create_var(torch.zeros(self.hidden_size), False))
						else:
							prev_mol_id = template_node.children[n-1].id
							prev_vectors_t.append(h_pred[(rxn_id, prev_mol_id)])
				prev_vectors_t = torch.stack(prev_vectors_t, dim=0)
				latent_vectors_t = torch.stack(latent_vectors_t, dim = 0)
				product_vectors_t = torch.stack(product_vectors_t, dim=0)
				#template_vectors_t = torch.stack(template_vectors_t, dim=0)
				template_vectors_t = self.t_embedding(create_var(torch.LongTensor(template_vectors_t)))
				context = attention(cur_enc_outputs_t, product_vectors_t)
				cur_input = torch.cat([prev_vectors_t,latent_vectors_t, product_vectors_t, template_vectors_t, context], dim=1)
				cur_output = self.W_reactant(cur_input)
				cur_output = nn.ReLU()(cur_output)
				#print(cur_output)



				for i in range(len(rxn_tem_mols)):
					rxn_id, template_id, mol_id = rxn_tem_mols[i]
					#if (rxn_id, mol_id) in h.keys():
					#	debug.append((rxn_id, mol_id))
						#exit(1)
					h_pred[(rxn_id, mol_id)] = cur_output[i]
		starting_react_vecs=[]
		starting_react_ids =[]
		latents =[]
		product_vecs = []
		template_vecs =[]
		for rxn_id, mol_id in leaf_nodes.keys():
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			template_id = molecule_nodes[mol_id].parents[0].id
			product_id = molecule_nodes[mol_id].parents[0].parents[0].id
			starting_react_vecs.append(h_pred[(rxn_id, mol_id)])
			starting_react_ids.append(leaf_nodes[(rxn_id, mol_id)])
			latents.append(latent_vecs[rxn_id])
			template_vecs.append(l_pred[(rxn_id, template_id)])
			product_vecs.append(h_pred[(rxn_id, product_id)])
		latents = torch.stack(latents, dim = 0)
		template_vecs =torch.stack(template_vecs, dim=0)
		product_vecs = torch.stack(product_vecs, dim=0)
		starting_react_vecs = torch.stack(starting_react_vecs, dim=0)
		starting_react_vecs = torch.cat([latents, starting_react_vecs, template_vecs, product_vecs],dim=1)
		starting_react_vecs = self.W_starting_react(starting_react_vecs)
		starting_react_ids = create_var(torch.LongTensor(starting_react_ids))
		starting_react_loss = self.starting_react_loss(starting_react_vecs, starting_react_ids)
		starting_react_vecs = F.softmax(starting_react_vecs, dim=1)
		_, starting_react_vecs = starting_react_vecs.max(dim=-1)
		starting_react_correct = (starting_react_vecs == starting_react_ids).float().sum()
		#======================================================================

		tem_targets = []
		tem_pred = []
		for rxn_id in range(len(rxn_tree_batch)):
			template_nodes = rxn_tree_batch[rxn_id].template_nodes
			for tem_id, tem_node in enumerate(template_nodes):
				tem_targets.append(l_target[(rxn_id, tem_id)])
				tem_pred.append(logits_pred[(rxn_id,tem_id)])
		tem_pred = torch.stack(tem_pred,dim=0)
		tem_targets = create_var(torch.LongTensor(tem_targets))
		#print(tem_targets)
		template_loss = self.template_loss(tem_pred, tem_targets)
		output = F.softmax(tem_pred, dim=1)
		_, output = output.max(dim=-1)
			#print(outputs
		correct = (output == tem_targets).float().sum()


		return 0.0,template_loss/len(rxn_tree_batch), starting_react_loss/len(rxn_tree_batch), correct/tem_targets.shape[0], starting_react_correct/starting_react_vecs.shape[0]




class RXNDecoder(nn.Module):
	def __init__(self, hidden_size, latent_size, reactantDic, templateDic, molecule_embedding=None, template_embedding=None, mpn = None):
		super(RXNDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.reactantDic = reactantDic
		self.templateDic = templateDic

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.templateDic.size(), self.hidden_size)
		else:
			self.template_embedding = template_embedding
		#if fragment_embedding is None:
		#	self.fragment_embedding = nn.Embedding(self.fragmentDic.size(), self.hidden_size)
		#else:
		#	self.fragment_embedding = fragment_embedding

		if molecule_embedding is None:
			self.molecule_embedding = nn.Embedding(self.reactantDic.size(), self.hidden_size)
		else:
			self.molecule_embedding = molecule_embedding
		if mpn is None:
			self.mpn = MPN(self.hidden_size, 2)
		else:
			self.mpn = mpn

		self.molecule_distance_loss = nn.MSELoss(size_average = False)
		self.template_loss = nn.CrossEntropyLoss(size_average = False)
		self.molecule_label_loss = nn.CrossEntropyLoss(size_average = False) 

		self.W_root = nn.Linear(self.latent_size, self.hidden_size)
		self.W_template = nn.Linear(2 * self.hidden_size + self.latent_size, self.templateDic.size())
		self.W_reactant = nn.Linear(3 * self.hidden_size + self.latent_size, self.hidden_size)
		self.W_reactant_out = nn.Linear(2 * self.hidden_size + self.latent_size, self.hidden_size)
		self.W_label = nn.Linear(self.hidden_size, self.reactantDic.size())



		# update molecules
		self.gru = nn.GRU(3 * self.hidden_size + self.latent_size, self.hidden_size, dropout=0.5)
		self.gru_template = nn.GRU(1 * self.hidden_size + self.latent_size, self.hidden_size, dropout=0.5)


	def decode_many_time(self, latent_vector, encoder_outputs, n):
		results =[]
		for i in range(n):
			res1, res2 = self.decode(latent_vector, encoder_outputs)
			if res1 != None:
				return res1, res2
		return None, None


	def decode(self, latent_vector, encoder_outputs, prob_decode=True):
		#context_vector = attention(encoder_outputs, latent_vector)
		root_embedding = self.W_root(torch.cat([latent_vector], dim=1))
		root_embedding = nn.ReLU()(root_embedding)

		molecule_labels ={}
		template_labels ={}
		molecule_hs ={}
		template_hs ={}
		
		queue = deque([])
		molecule_nodes =[]
		template_nodes =[]
		molecule_counter = 0
		template_counter = 0
		tree_root = MoleculeNode("", molecule_counter)
		molecule_hs[molecule_counter] = root_embedding
		molecule_nodes.append(tree_root)
		molecule_labels[molecule_counter] = self.reactantDic.get_index("unknown")
		molecule_counter += 1


		# predict template
		product_vector = molecule_hs[0]
		context = attention(encoder_outputs, product_vector)
		prev_xs = torch.cat([latent_vector, context], dim=1).unsqueeze(0)
		os, hs = self.gru_template(prev_xs, product_vector.unsqueeze(0))
		hs = hs[0,:,:]

		context = attention(encoder_outputs, hs)
		logits = self.W_template(torch.cat([latent_vector, context, hs], dim=1))
		output = F.softmax(logits, dim=1)
		if prob_decode:
			template_type = torch.multinomial(output[0], 1)
		else:
			_, template_type = torch.max(output, dim=1)
		template_node = TemplateNode("", template_counter)
		template_node.template_type = template_type
		template_node.parents.append(tree_root)
		tree_root.children.append(template_node)
		template_nodes.append(template_node)
		#template_vec = self.template_embedding(create_var(torch.LongTensor(template_type)))
		#l[template_id] = template_vec
		template_hs[template_counter] = hs
		template_labels[template_counter] = template_type
		template_counter += 1

		n_reactants = self.templateDic.get_n_reacts(template_type.item())
		product_id = template_node.parents[0].id
		for n in range(n_reactants):
			if n == 0:
				temp_id = create_var(torch.LongTensor(template_node.template_type))
				pre_xs = self.template_embedding(temp_id)
				pre_hs = template_hs[template_node.id]
			else:
				prev_mol_id = template_node.children[n-1].id
				mol_label =molecule_labels[prev_mol_id]
				#print(mol_label)
				pre_xs = self.molecule_embedding(mol_label)
				pre_hs = molecule_hs[prev_mol_id]
			context = attention(encoder_outputs, pre_hs)
			pre_xs = torch.cat([latent_vector, context, pre_xs, molecule_hs[product_id]], dim=1)
			os, hs = self.gru(pre_xs.unsqueeze(0), pre_hs.unsqueeze(0))
			hs = hs[0,:,:]
			molecule_hs[molecule_counter] = hs
			context = attention(encoder_outputs, hs)
			input = torch.cat([latent_vector, context, hs], dim = 1)
			output = nn.ReLU()(self.W_reactant_out(input))
			output = self.W_label(output)
			output = F.softmax(output, dim=1)
			#label = output.max(dim=-1)
			
			if prob_decode:
				label = torch.multinomial(output[0], 1)
			else:
				_, label = torch.max(output, dim=1)
			molecule_labels[molecule_counter] = label
			reactant_node = MoleculeNode("", molecule_counter)
			reactant_node.parents.append(template_node)
			template_node.children.append(reactant_node)
			queue.append(reactant_node)
			molecule_counter+=1
		count =1
		while len(queue)> 0:
			#print(queue)
			count +=1
			if count > 20:
				return None, None
			cur_molecule_node = queue.popleft()
			molecule_nodes.append(cur_molecule_node)
			template_node = cur_molecule_node.parents[0]
			pre_molecule_node = template_node.parents[0]

			temp_id = template_node.id
			pre_molecule_id = pre_molecule_node.id

			cur_molecule_vec = molecule_hs[cur_molecule_node.id]
			pre_molecule_vec = molecule_hs[pre_molecule_node.id]
			template_vec = template_hs[template_node.id]

			context = attention(encoder_outputs, cur_molecule_vec)
			input = torch.cat([latent_vector, context, cur_molecule_vec], dim=1)
			output = nn.ReLU()(self.W_reactant_out(input))
			output = self.W_label(output)
			output = F.softmax(output, dim=1)
			if prob_decode:
				output = torch.multinomial(output[0], 1)
			else:
				_, output = torch.max(output, dim=1)
			if output.item()== self.reactantDic.size()-1:
				# continue to perform reaction
				# predict tempalte
				context = attention(encoder_outputs, cur_molecule_vec)
				pre_xs  = torch.cat([latent_vector, context], dim=1).unsqueeze(0)
				os, hs = self.gru_template(pre_xs, cur_molecule_vec.unsqueeze(0))
				hs = hs[0,:,:]

				context = attention(encoder_outputs, hs)
				logits = self.W_template(torch.cat([latent_vector, context, hs], dim=1))
				output = F.softmax(logits, dim=1)
				#template_type = torch.multinomial(output[0], 1)
				_, template_type = torch.max(output, dim=1)

				template_node = TemplateNode("", template_counter)
				template_node.parents.append(cur_molecule_node)
				template_node.template_type = template_type
				cur_molecule_node.children.append(template_node)
				template_nodes.append(template_node)
				template_labels[template_counter] = template_type
				template_hs[template_counter] = hs
				n_reactants = self.templateDic.get_n_reacts(template_type.item())
				for n in range(n_reactants):
					if n==0:
						temp_id = create_var(torch.LongTensor(template_node.template_type))
						pre_xs = self.template_embedding(temp_id)
						pre_hs = template_hs[template_node.id]
					else:
						prev_mol_id = template_node.children[n-1].id
						mol_label =molecule_labels[prev_mol_id]
						#print(mol_label)
						pre_xs = self.molecule_embedding(mol_label)
						pre_hs = molecule_hs[prev_mol_id]
					context = attention(encoder_outputs, pre_hs)
					pre_xs = torch.cat([latent_vector, context, pre_xs, cur_molecule_vec], dim=1)
					os, hs = self.gru(pre_xs.unsqueeze(0), pre_hs.unsqueeze(0))
					hs = hs[0,:,:]
					molecule_hs[molecule_counter] = hs
					context = attention(encoder_outputs, hs)
					input = torch.cat([latent_vector, context, hs], dim = 1)
					output = nn.ReLU()(self.W_reactant_out(input))
					output = self.W_label(output)
					output = F.softmax(output, dim=1)
					
					if prob_decode:
						label = torch.multinomial(output[0], 1)
					else:
						_, label = torch.max(output, dim=1)
					molecule_labels[molecule_counter] = label
					reactant_node = MoleculeNode("", molecule_counter)
					reactant_node.parents.append(template_node)
					template_node.children.append(reactant_node)
					#print(n, molecule_labels)

					queue.append(reactant_node)
					molecule_counter+=1
				template_counter +=1
			else:
				cur_molecule_node.reactant_id = output[0].item()

		#print("dkm", molecule_labels, template_labels)
		# generate the final product
		#print(molecule_labels, template_labels)
		#for node in molecule_nodes:
		#	print(node.id)
		#for node in template_nodes:
		#	print(node.id, node.template_type)


		node2smiles ={}
		root = template_nodes[0]
		queue = deque([root])
		visited = set([root.id])
		root.depth = 0
		order ={}
		order[0] = [root.id]
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
		max_depth = len(order) - 1
		for t in range(max_depth, -1, -1):
			for template_id in order[t]:
				template_node = template_nodes[template_id]
				template_type = template_node.template_type.item()
				template = self.templateDic.get_template(template_type)
				reactants = []
				for reactant_node in template_node.children:
					if len(reactant_node.children) == 0:
						reactant = self.reactantDic.get_reactant(reactant_node.reactant_id)
						reactants.append(reactant)
						node2smiles[reactant_node.id] = reactant
					else:
						reactant = node2smiles[reactant_node.id]
						reactants.append(reactant)
				possible_templates = reverse_template(template)
				possible_products = []
				reacts = [Chem.MolFromSmiles(reactant) for reactant in reactants]
				for template in possible_templates:
					rxn = rdChemReactions.ReactionFromSmarts(template)
					AllChem.SanitizeRxn(rxn)
					#for reacts in possible_reacts:
					products = rxn.RunReactants(reacts)
					if len(products) > 0:
						n = len(products)
						for i in range(n):
							product = products[i]
							possible_products.append(product[0])
				if len(possible_products) > 0:
					product_id = template_node.parents[0].id
					node2smiles[product_id] = Chem.MolToSmiles(possible_products[0])
				else:
					success = False
					return None, None
		str_reactions=[]
		for t in range(len(order)):
			for template_id in order[t]:
				template_node = template_nodes[template_id]
				template_type = template_node.template_type.item()
				template = self.templateDic.get_template(template_type)
				reactants =[]
				for reactant_node in template_node.children:
					reactant = node2smiles[reactant_node.id]
					reactants.append(reactant)
				product_id = template_node.parents[0].id
				product = node2smiles[product_id]
				reactants =".".join(reactants)
				reaction = "*".join([product, reactants, template])
				str_reactions.append(reaction)
		str_reactions = " ".join(str_reactions)
		return node2smiles[0], str_reactions


	def forward(self, rxn_tree_batch, latent_vectors, encoder_outputs):
		# intiualize sth
		template_loss = 0
		template_acc = 0
		n_templates = 0
		molecule_distance_loss = 0
		n_molecules = 0
		molecule_label_loss = 0
		label_acc = 0

		orders =[]
		B = len(rxn_tree_batch)
		for rxn_tree in rxn_tree_batch:
			order = get_template_order(rxn_tree)
			orders.append(order)
		max_depth = max([len(order) for order in orders])

		target_molecules =[]
		target_templates =[]
		molecule_labels={}

		for rxn_id in range(len(rxn_tree_batch)):
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			template_nodes = rxn_tree_batch[rxn_id].template_nodes
			for ind, molecule_node in enumerate(molecule_nodes):
				#target_molecules.append(molecule_node.smiles)
				if len(molecule_node.children)==0:
					molecule_labels[(rxn_id, ind)] = self.reactantDic.get_index(molecule_node.smiles)
				else:
					molecule_labels[(rxn_id, ind)] = self.reactantDic.get_index("unknown")
			for template_node in template_nodes:
				target_templates.append(self.templateDic.get_index(template_node.template))
		#target_mol_embeddings = self.mpn(target_molecules)

		
		o_target ={}
		l_target ={}
		h_pred ={}
		l_pred ={}
		o_pred = {}
		logits_pred={}
		template_hids={}
		template_outs={}
		#i=0
		#for rxn_id in range(len(rxn_tree_batch)):
		#	for j in range(len(rxn_tree_batch[rxn_id].molecule_nodes)):
		#		o_target[(rxn_id, j)] = target_mol_embeddings[i]
		#		i=i+1
		i = 0
		for rxn_id in range(len(rxn_tree_batch)):
			for j in range(len(rxn_tree_batch[rxn_id].template_nodes)):
				l_target[(rxn_id, j)] = target_templates[i]
				i+=1

		#context_vectors = attention(encoder_outputs, latent_vectors)
		root_embeddings = self.W_root(torch.cat([latent_vectors], dim=1))
		root_embeddings = nn.ReLU()(root_embeddings)

		

		for rxn_id in range(B):
			h_pred[(rxn_id, 0)] = root_embeddings[rxn_id]
			o_pred[(rxn_id, 0)]= root_embeddings[rxn_id]

		for t in range(max_depth):
			tem_E ={}
			template_targets =[]
			template_hs ={}
			# for tracking
			template_ids =[]
			rxn_ids =[]
			template2reactants={}
			for i, order in enumerate(orders):
				if len(order) > t:
					template_ids.extend(order[t])
					rxn_ids.extend([i]*len(order[t]))
			product_vectors_t =[]
			latent_vectors_t =[]
			cur_enc_outputs=[]
			n_reactants =[]
			tem_targets =[]
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
				product_id = template_node.parents[0].id
				product_vectors_t.append(h_pred[(rxn_id, product_id)])
				latent_vectors_t.append(latent_vectors[rxn_id])
				cur_enc_outputs.append(encoder_outputs[rxn_id])
				tem_targets.append(self.templateDic.get_index(template_node.template))
				#template2reactants.append(len(template_node.children))
				n_reactants.append(len(template_node.children))
			#prev_hs = torch.stack(prev_hs, dim=0).unsqueeze(0)
			#prev_xs = torch.stack(prev_xs, dim=0)
			#latent_vectors_t = torch.stack(latent_vectors_t, dim = 0)
			#product_vectors_t = torch.stack(product_vectors_t, dim=0)
			
			
			#prev_xs = torch.cat([prev_xs, latent_vectors_t, product_vectors_t, template_vectors_t], dim=1).unsqueeze(0)
			#os, hs = self.gru(prev_xs, prev_hs)
			#os, hs = os[0,:,:], hs[0,:,:]
			#prev_hs = torch.stack(product_vectors_t, dim=0)
			product_vectors_t = torch.stack(product_vectors_t, dim=0)
			latent_vectors_t = torch.stack(latent_vectors_t, dim=0)
			context = attention(cur_enc_outputs, product_vectors_t)
			prev_xs = torch.cat([latent_vectors_t, context], dim=1).unsqueeze(0) 
			os, hs = self.gru_template(prev_xs, product_vectors_t.unsqueeze(0))
			hs = hs[0,:,:]
			#hs = hs[0,:,:]
			#print(hs.size())
			i=0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				template_hs[(rxn_id, template_id)] = hs[i]
				i+=1

			context = attention(cur_enc_outputs, hs)
			logits = self.W_template(torch.cat([latent_vectors_t, context, hs], dim=1))
			#print("logits", logits)
			#output = F.gumbel_softmax(logits, tau=0.5, hard=True, eps=1e-10, dim=-1)
			#_, ind = output.max(dim=-1)
			#template_vecs = create_var(torch.LongTensor(ind))
			#template_vecs = self.template_embedding(template_vecs)
			#i=0
			#for template_id, rxn_id in zip(template_ids, rxn_ids):
			#	tem_E[(rxn_id, template_id)] = template_vecs[i]
			#	i+=1

			# for target
			tem_vecs = self.template_embedding(create_var(torch.LongTensor(tem_targets)))
			i=0
			
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				tem_E[(rxn_id, template_id)] = tem_vecs[i]
				i+=1





			i=0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				logits_pred[(rxn_id, template_id)] = logits[i]
				i+=1

			tem_targets = create_var(torch.LongTensor(tem_targets))
			template_loss += self.template_loss(logits, tem_targets)
			output = F.softmax(logits, dim=1)
			#print(output)
			_, output = output.max(dim=-1)
			template_acc += (output == tem_targets).float().sum()
			n_templates += tem_targets.shape[0]
			#output = F.gumbel_softmax(logits, tau=0.5, hard=True, eps=1e-10, dim=-1)
			#output = F.softmax(logits, dim=1)
			#_, ind = output.max(dim=-1)

			#template_vecs = create_var(torch.LongTensor(ind.cpu()))

			#template_vecs = self.template_embedding(template_vecs)
			i = 0
			#for template_id, rxn_id in zip(template_ids, rxn_ids):
				#l[(rxn_id, template_id)] = template_vecs[i]
			#	l_pred[(rxn_id, template_id)] = template_vecs[i]
			#	i+=1
			max_n_reactants = max(n_reactants)


			for n in range(max_n_reactants):
				rxn_tem_mols= []
				prev_hs = []
				prev_xs = []
				latent_vectors_t = []
				product_vectors_t = []
				template_vectors_t = []
				encoder_outputs_t =[]
				mol_targets=[]
				target_labels=[]
				mol_ids =[]
				mol_E={}

				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						if n == 0:
							continue
						else:
							#prev_mol_id = template_node.children[n-1].smiles
							molecule_node = template_node.children[n-1]
							if len(molecule_node.children) == 0:
								id = self.reactantDic.get_index(template_node.children[n-1].smiles)
							else:
								id = self.reactantDic.get_index("unknown")
							mol_ids.append(id)

				mol_ids = create_var(torch.LongTensor(mol_ids))
				embeddings = self.molecule_embedding(mol_ids)
				i=0
				
				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						if n == 0:
							continue
						else:
							prev_mol_id = template_node.children[n-1].id
							mol_E[(rxn_id, prev_mol_id)] = embeddings[i]
							i+=1
				
				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						mol_id = template_node.children[n].id
						mol_targets.append(template_node.children[n].smiles)
						product_id = template_node.parents[0].id
						rxn_tem_mols.append((rxn_id, template_id, mol_id))
						target_labels.append(molecule_labels[(rxn_id, mol_id)])

						latent_vectors_t.append(latent_vectors[rxn_id])
						product_vectors_t.append(h_pred[(rxn_id, product_id)])
						template_vectors_t.append(l_target[(rxn_id, template_id)])
						encoder_outputs_t.append(encoder_outputs[rxn_id])
						#mol_targets.append(o_target[(rxn_id, mol_id)])
						if n==0:
							#prev_xs.append(create_var(torch.zeros(self.hidden_size), False))
							#prev_hs.append(create_var(torch.zeros(self.hidden_size), False))
							prev_xs.append(tem_E[(rxn_id, template_id)])
							prev_hs.append(template_hs[(rxn_id, template_id)])
						else:
							prev_mol_id = template_node.children[n-1].id
							prev_xs.append(mol_E[(rxn_id, prev_mol_id)])
							prev_hs.append(h_pred[(rxn_id, prev_mol_id)])


				prev_hs = torch.stack(prev_hs, dim=0)
				prev_xs = torch.stack(prev_xs, dim=0)
				latent_vectors_t = torch.stack(latent_vectors_t, dim = 0)
				product_vectors_t = torch.stack(product_vectors_t, dim=0)
				#template_vectors_t = self.template_embedding(create_var(torch.LongTensor(template_vectors_t)))
				context = attention(encoder_outputs_t, prev_hs)
				prev_xs = torch.cat([latent_vectors_t, context, prev_xs, product_vectors_t], dim=1)

				os, hs = self.gru(prev_xs.unsqueeze(0), prev_hs.unsqueeze(0))# pre_hs-> product
				hs, os = hs[0,:,:], os[0,:,:]
				for i in range(len(rxn_tem_mols)):
					rxn_id, template_id, mol_id = rxn_tem_mols[i]
					h_pred[(rxn_id, mol_id)] = hs[i]
					o_pred[(rxn_id, mol_id)] = os[i]
				context = attention(encoder_outputs_t, hs)
				mol_preds = nn.ReLU()(self.W_reactant_out(torch.cat([latent_vectors_t, context, hs], dim=1)))
				#mol_targets = self.mpn(mol_targets)
				#molecule_distance_loss += self.molecule_distance_loss(mol_preds, mol_targets)
				n_molecules += mol_preds.shape[0]

				pred_labels = self.W_label(mol_preds)
				target_labels = create_var(torch.LongTensor(target_labels))
				molecule_label_loss += self.molecule_label_loss(pred_labels, target_labels)
				pred_labels = F.softmax(pred_labels, dim=1)
				_, pred_labels = pred_labels.max(dim=-1)
				label_acc += (pred_labels==target_labels).float().sum()



				#mol_targets=[]
				

				#for i in range(len(rxn_tem_mols)):
				#	rxn_id, template_id, mol_id = rxn_tem_mols[i]
				#	h_pred[(rxn_id, mol_id)] = hs[i]
					#o_pred[(rxn_id, mol_id)] = os[i]
		'''
		tem_targets = []
		tem_pred = []
		for rxn_id in range(len(rxn_tree_batch)):
			template_nodes = rxn_tree_batch[rxn_id].template_nodes
			for tem_id, tem_node in enumerate(template_nodes):
				tem_targets.append(l_target[(rxn_id, tem_id)])
				tem_pred.append(logits_pred[(rxn_id,tem_id)])
		tem_pred = torch.stack(tem_pred,dim=0)
		tem_targets = create_var(torch.LongTensor(tem_targets))
		#print(tem_targets)
		template_loss = self.template_loss(tem_pred, tem_targets)
		# gumbel sampling
			#output = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1)
			#_, ind = output.max(dim=-1)

		#softmax
		output = F.softmax(tem_pred, dim=1)
		_, output = output.max(dim=-1)
			#print(outputs
		template_acc = (output == tem_targets).float().sum()


		
		mol_targets=[]
		h_preds_t=[]
		o_preds_t =[]
		encoder_outputs_t=[]
		latent_vectors_t=[]

		for rxn_id in range(len(rxn_tree_batch)):
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			for mol_id, mol_node in enumerate(molecule_nodes):
				mol_targets.append(o_target[(rxn_id, mol_id)])
				h_preds_t.append(h_pred[(rxn_id, mol_id)])
				o_preds_t.append(o_pred[(rxn_id, mol_id)])
				encoder_outputs_t.append(encoder_outputs[rxn_id])
				latent_vectors_t.append(latent_vectors[rxn_id])
		h_preds_t = torch.stack(h_preds_t, dim=0)
		o_preds_t = torch.stack(o_preds_t, dim=0)
		latent_vectors_t = torch.stack(latent_vectors_t, dim=0)
		context = attention(encoder_outputs_t, h_preds_t)

		mol_preds = nn.ReLU()(self.W_reactant_out(torch.cat([latent_vectors_t, context, o_preds_t], dim=1)))
		mol_targets = torch.stack(mol_targets,dim=0)
		molecule_distance_loss = self.molecule_distance_loss(mol_preds, mol_targets)



		target_labels=[]
		pred_labels =[]
		for rxn_id in range(len(rxn_tree_batch)):
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			for mol_id, mol_node in enumerate(molecule_nodes):
				target_labels.append(molecule_labels[(rxn_id, mol_id)])
		target_labels = create_var(torch.LongTensor(target_labels))
		pred_labels = self.W_label(mol_preds)
		molecule_label_loss = self.molecule_label_loss(pred_labels, target_labels)
		pred_labels = F.softmax(pred_labels, dim=1)
		_, pred_labels = pred_labels.max(dim=-1)
		label_acc = (pred_labels==target_labels).float().sum()
		'''		

		return 0.1 * molecule_distance_loss/n_molecules, template_loss/n_templates, molecule_label_loss/n_molecules, template_acc/n_templates, label_acc/n_molecules




























