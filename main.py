
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


from evaluate import Evaluator

from reaction_utils import read_multistep_rxns
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed


def reaction_tree_generation(model, n, args):
	prob_decode = True
	batch_size = args['batch_size']
	trees=[]
	mean_xs = []
	for i in range(batch_size):
		
		mean_x = torch.randn(model.latent_size)
		mean_xs.append(mean_x)
		generated_tree=model.fragment_decoder.decode(mean_x.unsqueeze(0), prob_decode=True)
		trees.append(generated_tree)
		print("generating:", i)
	mean_xs = torch.stack(mean_xs, dim=0)
	g_encoder_outputs, g_root_vecs = model.fragment_encoder(trees)
	#mean_ys_ = model.ft_mean_qxy(model.ft_mean_qx(g_root_vecs))
	mean_ys = model.ft_mean_qxy(mean_xs)
	#print(mean_ys.size(), mean_ys_.size())
	#exit(1)

	mean_ys = model.ft_mean_qxy(mean_xs)
	count = 0
	for i in range(batch_size):
		if trees[i] is None:
			continue
		tmp = mean_ys[i]
		#product, reaction = model.rxn_decoder.decode_many_time(torch.stack([tmp], dim=0), [g_encoder_outputs[i]], n)
		for t in range(n):
			product, reaction = model.rxn_decoder.decode(torch.stack([tmp], dim=0), [g_encoder_outputs[i]], True)
			if product != None:
				break
		print(i, product)
		if product != None:
			count +=1
	return count


def generation(data_pair, model, args):
	prob_decode = True
	latent_size = args['latent_size']
	ft_tree, rxn_tree = data_pair[0], data_pair[1]
	ft_trees = [ft_tree]
	rxn_trees = [rxn_tree]
	set_batch_nodeID(ft_trees, model.fragment_vocab)
	encoder_outputs, root_vecs = model.fragment_encoder(ft_trees)
	root_vec_rxns = model.rxn_encoder(rxn_trees)

	ft_mean = model.FT_mean(root_vecs)
	ft_log_var = -torch.abs(model.FT_var(root_vecs))
	epsilon = create_var(torch.randn(1, latent_size), False) * 0.01
	#ft_mean = ft_mean + torch.exp(ft_log_var/2) * epsilon

	rxn_mean = model.RXN_mean(root_vec_rxns)
	rxn_log_var = -torch.abs(model.RXN_var(root_vec_rxns))
	epsilon = create_var(torch.randn(1, latent_size), False) * 0.01
	#rxn_mean = rxn_mean + torch.exp(rxn_log_var/2) * epsilon


	# decode the tree
	for i in range(20):
		x = ft_mean + torch.exp(ft_log_var/2) * epsilon
		y = rxn_mean + torch.exp(rxn_log_var/2) * epsilon
		generated_tree=model.fragment_decoder.decode(x, prob_decode)
		g_encoder_output, g_root_vec = model.fragment_encoder([generated_tree])
		product, reactions = model.rxn_decoder.decode(y, g_encoder_output, prob_decode)

		if product != None:
			return product, reactions
	return None, None
	#encoder_output, root_vec = model.fragment_encoder([generated_tree])


def validate(data_pairs, model, args):
	#model.eval()
	beta = args['beta']
	batch_size = args['batch_size']
	dataloader = DataLoader(data_pairs, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)

	total_pred_acc =0
	total_stop_acc = 0
	total_template_loss = 0
	total_template_acc = 0
	total_molecule_distance_loss =0
	#total_molecule_label_loss = 0
	total_label_acc =0
	total_pred_loss=0
	total_stop_loss =0
	total_template_loss = 0
	total_molecule_label_loss = 0
	total_loss = 0

	with torch.no_grad():
		for it, batch in enumerate(dataloader):

			t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, epsilon_std=0.01)
			total_pred_acc += pred_acc
			total_stop_acc += stop_acc
			total_template_acc += template_acc
			total_label_acc += label_acc
			total_pred_loss += pred_loss
			total_stop_loss += stop_loss
			total_template_loss += template_loss
			total_molecule_label_loss += molecule_label_loss
			#print(it, pred_acc, stop_acc, template_acc, label_acc)
		#total

	print("*** pred loss: ",total_pred_loss.item()/len(dataloader), "pred acc:", total_pred_acc/len(dataloader))
	print("*** stop loss: ",total_stop_loss.item()/len(dataloader), "stop acc:", total_stop_acc/len(dataloader))
	#print("*** molecule dsitance loss: ",molecule_distance_loss.item())

	print("*** template loss: ",total_template_loss.item()/len(dataloader), "template acc:", total_template_acc/len(dataloader))
	print("*** label loss: ",total_molecule_label_loss.item()/len(dataloader), "label acc:", total_label_acc/len(dataloader))
	#print("*** total loss:", t_loss.item()- beta * kl_loss.item())
	return t_loss - beta * kl_loss


def schedule(counter, M):
	x = counter/(2*M)
	if x > M:
		return 1.0
	else:
		return 1.0 * x/M

def abschedule(counter, M):
	x = counter/M
	a = x/M
	b = 1- a
	return a, b

def train(data_pairs, model,args):
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	print("You are working on:",device, "n of data:", len(data_pairs))

	#model.cuda()

	import pickle
	n_pairs = len(data_pairs)
	ind_list = [i for i in range(n_pairs)]
	random.shuffle(ind_list)
	#print(ind_list)
	with open('ind_list2.txt', 'rb') as f:
		ind_list = pickle.load(f)
	#with open('ind_list.txt', 'wb') as f:
	#	pickle.dump(ind_list, f)
	#print(ind_list)
	#random.shuffle(data_pairs)
	data_pairs = [data_pairs[i] for i in ind_list]
	#print("Number of data pairs:", n_pairs)
	lr = args['lr']
	batch_size = args['batch_size']
	beta = args['beta']
	val_pairs = data_pairs[:1000]
	train_pairs = data_pairs[1000:-1]
	print("trainng size:", len(train_pairs))
	print("valid size:", len(val_pairs))
	#for i in range(n_pairs):
	#	print("data point ",i)
	#	fgm_tree, rxn_tree = data_pairs[i][0], data_pairs[i][1]
	#	print("---> ", fgm_tree.smiles)
	#	print("---> ", rxn_tree.molecule_nodes[0].smiles)
	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
	tr_rec_loss_list = []
	tr_kl_loss_list=[]
	beta_list=[]
	M = 100

	counter = 0

	for epoch in range(args['epochs']):
		#if beta < 1.0 and epoch > 10:
		#	beta += 0.1
		#beta = 1.0
		#beta_list.append(beta)
		random.shuffle(train_pairs)

		dataloader = DataLoader(train_pairs, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)
		total_loss = 0
		total_pred_loss=0
		total_stop_loss =0
		total_kl_loss =0
		total_pred_acc =0
		total_stop_acc = 0
		total_template_loss = 0
		total_template_acc = 0
		total_molecule_distance_loss =0
		total_molecule_label_loss = 0
		total_label_acc =0
		for it, batch in enumerate(dataloader):
			#print(epoch, it, len(dataloader))
			#if epoch < 20:
			#	beta = schedule(counter, M)
			#else:
			#	beta = args['beta']
			#if epoch > 10:
				#a,b = abschedule(counter, M)
			a,b = 0.5, 0.5
			#else:
			#	a,b = 0.0, 1.0
			counter +=1
			model.zero_grad()
			t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, a, b)
			#total_loss, pred_loss, stop_loss, template_loss, molecule_distance_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss

			#nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
			t_loss.backward()
			optimizer.step()
			#scheduler.step()
			total_loss += t_loss
			total_pred_loss += pred_loss
			total_stop_loss += stop_loss
			total_kl_loss += kl_loss
			total_pred_acc += pred_acc
			total_stop_acc += stop_acc
			total_template_loss += template_loss
			total_template_acc += template_acc
			total_molecule_distance_loss += molecule_distance_loss
			total_molecule_label_loss += molecule_label_loss
			total_label_acc += label_acc
			
				

		
		print("*******************Epoch", epoch, "******************", counter, beta)
		val_loss = validate(val_pairs, model, args)
		print("---> pred loss:", total_pred_loss.item()/len(dataloader), "pred acc:", total_pred_acc/len(dataloader))
		print("---> stop loss:", total_stop_loss.item()/len(dataloader), "stop acc:", total_stop_acc/len(dataloader))
		print("---> template loss:", total_template_loss.item()/len(dataloader), "tempalte acc:", total_template_acc.item()/len(dataloader))
		#print("---> molecule distance loss:", total_molecule_distance_loss.item()/len(dataloader))
		print("---> molecule label loss:", total_molecule_label_loss.item()/len(dataloader), "molecule acc:", total_label_acc.item()/len(dataloader))
		print("---> kl loss:", total_kl_loss.item()/len(dataloader))
		print("---> reconstruction loss:", total_loss.item()/len(dataloader)-beta * total_kl_loss.item()/len(dataloader))
		
		#if epoch %5 != 0:
		#	continue
		#t_rec_loss = total_loss.item()/len(dataloader) - beta * total_kl_loss.item()/len(dataloader)
		#t_kl_loss = total_kl_loss.item()/len(dataloader)
		#tr_rec_loss_list.append(t_rec_loss)
		#tr_kl_loss_list.append(t_kl_loss)


		
		if (epoch+1) %10 ==0:
			torch.save(model.state_dict(),args['datasetname']+ "_" + "vae_iter-{}.npy".format(epoch+1))
			print("saving file:", args['datasetname']+ "_" + "vae_iter-{}.npy".format(epoch+1))

			#checkpoint = torch.load(args['datasetname']+ "_" + "vae_iter-{}.npy".format(epoch + 1), map_location=device)
			#model1.load_state_dict(checkpoint)
			#print("*************************************")
			#val_loss = validate(val_pairs, model1, args)
			#print("*************************************")
	#for t in range(args['epochs']):
	#	print("iteration ", t, "reconstruction loss:", tr_rec_loss_list[t], "KL loss: ", tr_kl_loss_list[t], "beta:", beta_list[t])


def max_degree(fgm_tree):
	degrees=[]
	for node in fgm_tree.nodes:
		degrees.append(len(node.neighbors))
	return max(degrees)


#draw()
#exit(1)
is_training = False
data_filename ="data2.txt"

#routes, scores = read_multistep_rxns("synthetic_routes_uspto_qed_filtered.txt")
routes, scores = read_multistep_rxns(data_filename)
#routes=routes[:200]
rxn_trees = [ReactionTree(route) for route in routes]

molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]
reactants = extract_starting_reactants(rxn_trees)
templates, n_reacts = extract_templates(rxn_trees)

reactantDic = StartingReactants(reactants)
templateDic = Templates(templates, n_reacts)

	#print(ind_list)

n_pairs = len(routes)
ind_list = [i for i in range(n_pairs)]

#random.shuffle(ind_list)
#import pickle
#with open('ind_list2.txt', 'rb') as f:
#	ind_list = pickle.load(f)
#with open('ind_list.txt', 'wb') as f:
#	pickle.dump(ind_list, f)

#valid_inds = [ind_list[i] for i in range(2000)]

fgm_trees = [FragmentTree(rxn_trees[i].molecule_nodes[0].smiles) for i in ind_list]
rxn_trees = [rxn_trees[i] for i in ind_list]
max_degrees =[]
	#print(d)
#for i in range(5):
#	data_pairs.append((fgm_trees[i], rxn_trees[i]))
data_pairs=[]
for fgm_tree, rxn_tree in zip(fgm_trees, rxn_trees):
	data_pairs.append((fgm_tree, rxn_tree))

cset=set()
for fgm_tree in fgm_trees:
	for node in fgm_tree.nodes:
		cset.add(node.smiles)
cset = list(cset)
if is_training:
	fragmentDic = FragmentVocab(cset)#, filename ="fragmentvocab.txt")
	fragmentDic.save(data_filename+"_fragmentvocab.txt")
else:
	fragmentDic = FragmentVocab(cset, filename =data_filename+"_fragmentvocab.txt")






print("size of reactant dic:", reactantDic.size())
print("size of template dic:", templateDic.size())
print("size of fragment dic:", fragmentDic.size())

hidden_size = 200
latent_size = 50
depth = 2
batch_size = 32

args={}
args['hidden_size'] = hidden_size
args['latent_size'] = latent_size
args['depth'] = 2
args['epochs'] = 100
args['lr'] = 0.001
args['beta'] = 1.0
args['batch_size'] = batch_size
#args['datasetname'] = "uspto"
args['datasetname'] = "uspto"

if torch.cuda.is_available():
	#device = torch.device("cuda:1")
	device = torch.device("cuda")
	torch.cuda.set_device(1)
else:
	device = torch.device("cpu")

#print("current device:", torch.cuda.current_device())

print(args)
mpn = MPN(hidden_size, depth)

model = FTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, fragment_embedding=None, reactant_embedding=None, template_embedding=None)
if is_training:
	train(data_pairs, model,args)
else:
	print("loading model file")
	checkpoint = torch.load(args['datasetname']+ "_" + "vae_no_attention_iter_-{}.npy".format(100), map_location=device)
	model.load_state_dict(checkpoint)
	print("finished loading model...")
	#validate(data_pairs, model, args)
	count = 0

	evaluator = Evaluator(latent_size, model)
	#file1s=["qed0.txt","qed1.txt","qed2.txt","qed3.txt","qed4.txt","qed5.txt","qed6.txt","qed7.txt","qed8.txt","qed9.txt","qed10.txt"]
	#file2s = ["valid_reactions1.txt","valid_reactions2.txt"]
	#evaluator.novelty_and_uniqueness(["valid_reactions3.txt","valid_reactions4.txt"], rxn_trees)
	#evaluator.qualitycheck(rxn_trees, file2s)
	#evaluator.kde_plot(file2s, file1s)
	evaluator.validate_and_save(rxn_trees)
	exit(1)

	for i, data_pair in enumerate(data_pairs[:1000]):
		validate([data_pair], model, args)
		rxn = data_pair[1]
		target = rxn.molecule_nodes[0].smiles
		#for t in range(100):
		product, reaction = generation(data_pair, model, args)
		if product != None and product == target:
			count +=1
			print(i, count/(i+1), product)








