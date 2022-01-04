
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import QED
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from nnutils import create_var
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from vae import FTRXNVAE, set_batch_nodeID
from mpn import MPN,PP,Discriminator
import random
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import gzip
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import tqdm
import pandas as pd

#from rd_filters import rd_filters
import sys


class Evaluator(nn.Module):
	def __init__(self, latent_size, model):
		super(Evaluator, self).__init__()
		self.latent_size = latent_size
		self.model = model

	def decode_from_prior(self, ft_latent, rxn_latent, n, prob_decode=True):
		for i in range(n):
			#print("i:", i, n)
			generated_tree=self.model.fragment_decoder.decode(ft_latent, prob_decode=prob_decode)
			g_encoder_output, g_root_vec = self.model.fragment_encoder([generated_tree])
			product, reactions = self.model.rxn_decoder.decode(rxn_latent, g_encoder_output, prob_decode)
			if product != None:
				return product, reactions
		return None, None

	def novelty_and_uniqueness(self, files, rxn_trees):
		smiles_training_set=[]
		for rxn in rxn_trees:
			smiles = rxn.molecule_nodes[0].smiles
			smiles_training_set.append(smiles)


		smiles_training_set=set(smiles_training_set)
		training_size = len(smiles_training_set)
		count = 0
		total = 0
		valid_molecules =[]
		for file in files:
			with open(file, "r") as reader:
				lines =reader.readlines()
				for line in lines:
					total+=1
					elements = line.strip().split(" ")
					target = elements[0]
					reactions = elements[1:]
					if target not in smiles_training_set:
						valid_molecules.append(target)
						count+=1
						print(target, len(reactions))
			reader.close()
		print("novelty:", count, training_size, total, count/total)
		print("uniqueness:", len(set(valid_molecules)), len(valid_molecules), len(set(valid_molecules))/len(valid_molecules))


	def kde_plot(self, file1s, file2s, metric="qed"):
		bo_scores=[]
		smiles_list =[]
		for file in file2s:
			with open(file) as reader:
				lines = reader.readlines()
				for line in lines:
					line = line.strip()
					res = line.split(" ")
					smiles = res[0]
					if smiles not in smiles_list:
						smiles_list.append(smiles)
					else:
						continue
					try:
						if metric == "logp":
							score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
						if metric =="qed":
							score = get_qed_score(smiles)
						bo_scores.append(score)
						print(file, smiles, score)
					except:
						print("cannot parse:", smiles)
		smiles_list =[]
		p1= sns.kdeplot(bo_scores, shade=True, color='r', label='Bayesian Optimization', x='QED', clip=(-10,20))
		sampling_scores=[]
		count = 0
		for file in file1s:
			with open(file, "r") as reader:
				lines = reader.readlines()
				for line in lines:
					line = line.strip()
					res = line.split(" ")
					smiles = res[0]
					if smiles not in smiles_list:
						smiles_list.append(smiles)
					else:
						continue
					try:
						if metric == "logp":
							score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
						if metric =="qed":
							score = get_qed_score(smiles)
						sampling_scores.append(score)
						print(file, smiles, score)
					except:
						print("cannot parse:", smiles)
		np.random.shuffle(sampling_scores)
		limit = len(bo_scores)
		p1=sns.kdeplot(sampling_scores[:limit], shade=True, color='b',label='Random Sampling', clip=(-10,20))
		plt.xlabel('(a) QED', fontsize=18)
		plt.ylabel('', fontsize=18)
		plt.legend(loc='upper left')
		plt.show()
	def qualitycheck(self, rxns, files):
		smiles_list=[]
		for file in files:
			with open(file, "r") as reader:
				lines =reader.readlines()
				for line in lines:
					elements = line.strip().split(" ")
					target = elements[0]
					smiles_list.append(target)
		#print(smiles_list)
		num_cores = 4
		training_smiles = [rxn.molecule_nodes[0].smiles for rxn in rxns]

		p = Pool(mp.cpu_count())
		input_data = [(smi, f"MOL_{i}") for i, smi in enumerate(smiles_list)]
		training_data = [(smi, f"TMOL_{i}") for i, smi in enumerate(training_smiles)]
		
		alert_file_name = "alert_collection.csv"
		self.rf = rd_filters.RDFilters(alert_file_name)
		rules_file_path = "rules.json"
		rule_dict = rd_filters.read_rules(rules_file_path)
		rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
		rule_str = " and ".join(rule_list)
		print(f"Using alerts from {rule_str}", file=sys.stderr)
		self.rf.build_rule_list(rule_list)
		self.rule_dict = rule_dict
		#print(rule_list)

		trn_res = list(p.map(self.rf.evaluate, training_data))
		res = list(p.map(self.rf.evaluate, input_data))
		count1 = 0
		for re in trn_res:
			ok = re[2]
			if ok=="OK":
				count1 +=1
		norm = count1/len(trn_res)
		count2 = 0
		for re in res:
			ok = re[2]
			if ok=="OK":
				count2+=1
		ratio = count2/len(res)
		print(count2, len(res), ratio)
		print(ratio, norm, ratio/norm)



	def validate_and_save(self, train_rxn_trees, n=10000, output_file="generated_reactions.txt"):
		training_smiles =[]
		for tree in train_rxn_trees:
			smiles = tree.molecule_nodes[0].smiles
			training_smiles.append(smiles)
		validity = 0


		with open(output_file, "w") as writer:
			for i in range(n):
				#print(i)
				ft_latent = torch.randn(1,  self.latent_size)*0.1
				rxn_latent = torch.randn(1, self.latent_size)*0.1
				product, reactions = self.decode_from_prior(ft_latent, rxn_latent, 50)
				if product!= None:
					validity +=1
					print(i, validity/(i+1), "Product:",product, ", Reaction:", reactions)
					line = product + " " + reactions
					writer.write(line)
					writer.write("\n")

		print("validity: ", validity/n)



