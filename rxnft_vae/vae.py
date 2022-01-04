
import torch
import torch.nn as nn
from nnutils import create_var, attention
from ftencoder import FTEncoder
from ftdecoder import FTDecoder
from rxndecoder import RXNDecoder, RXNDecoder1
from rxnencoder import RXNEncoder
from mpn import MPN,PP,Discriminator


def set_batch_nodeID(ft_trees, ft_vocab):
	tot = 0
	for ft_tree in ft_trees:
		for node in ft_tree.nodes:
			node.idx = tot
			node.wid = ft_vocab.get_index(node.smiles)
			tot +=1
def log_Normal_diag(x, mean, log_var):
	log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
	return torch.mean(log_normal)


class FTRXNVAE(nn.Module):
	def __init__(self, fragment_vocab, reactant_vocab, template_vocab, hidden_size, latent_size, depth, fragment_embedding=None, reactant_embedding=None, template_embedding=None):
		super(FTRXNVAE, self).__init__()
		self.fragment_vocab = fragment_vocab
		self.reactant_vocab = reactant_vocab
		self.template_vocab = template_vocab
		self.depth = depth

		#print(self.fragment_vocab.vmap)
		#print(self.reactant_vocab.vmap)
		#print(self.template_vocab.vmap)

		self.hidden_size = hidden_size
		self.latent_size = latent_size

		if fragment_embedding is None:
			self.fragment_embedding = nn.Embedding(self.fragment_vocab.size(), hidden_size)
		else:
			self.fragment_embedding = fragment_embedding

		if reactant_embedding is None:
			self.reactant_embedding = nn.Embedding(self.reactant_vocab.size(), hidden_size)
		else:
			self.reactant_embedding = reactant_embedding

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.template_vocab.size(), hidden_size)
		else:
			self.template_embedding = template_embedding
		self.mpn = MPN(hidden_size, 2)


		self.fragment_encoder = FTEncoder(self.fragment_vocab, self.hidden_size, self.fragment_embedding)
		self.fragment_decoder = FTDecoder(self.fragment_vocab, self.hidden_size, self.latent_size, self.fragment_embedding)

		self.rxn_decoder = RXNDecoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.reactant_embedding, self.template_embedding, self.mpn)
		self.rxn_encoder = RXNEncoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.mpn, self.template_embedding)

		self.combine_layer = nn.Linear(2 *hidden_size, hidden_size)

		self.FT_mean = nn.Linear(hidden_size, int(latent_size))
		self.FT_var = nn.Linear(hidden_size, int(latent_size))

		self.RXN_mean = nn.Linear(hidden_size, int(latent_size))
		self.RXN_var = nn.Linear(hidden_size, int(latent_size))

	def encode(self, ftrxn_tree_batch):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)
		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		ft_mean = self.FT_mean(root_vecs)
		rxn_mean = self.RXN_mean(root_vecs_rxn)
		z_mean = torch.cat([ft_mean, rxn_mean], dim=1)
		return z_mean

	def forward(self, ftrxn_tree_batch, beta, a = 1.0, b = 1.0, epsilon_std=0.1):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)

		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		#root_vecs = torch.cat([root_vecs, root_vecs_rxn], dim=1)
		#root_vecs = self.combine_layer(input)
		#root_vecs = nn.ReLU()(root_vecs_rxn)
		ft_mean = self.FT_mean(root_vecs)
		ft_log_var = -torch.abs(self.FT_var(root_vecs))

		rxn_mean = self.RXN_mean(root_vecs_rxn)
		rxn_log_var = -torch.abs(self.RXN_var(root_vecs_rxn))

		z_mean = torch.cat([ft_mean, rxn_mean], dim=1)
		z_log_var = torch.cat([ft_log_var,rxn_log_var], dim=1)
		kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

		
		epsilon = create_var(torch.randn(batch_size, int(self.latent_size)), False)*epsilon_std
		ft_vec = ft_mean + torch.exp(ft_log_var / 2) * epsilon

		epsilon = create_var(torch.randn(batch_size, int(self.latent_size)), False)*epsilon_std
		rxn_vec = rxn_mean + torch.exp(rxn_log_var / 2) * epsilon

		pred_loss, stop_loss, pred_acc, stop_acc = self.fragment_decoder(ft_trees, ft_vec)
		molecule_distance_loss, template_loss, molecule_label_loss, template_acc, label_acc = self.rxn_decoder(rxn_trees, rxn_vec, encoder_outputs)
		
		rxn_decoding_loss = template_loss  + molecule_label_loss# + molecule_distance_loss
		fragment_decoding_loss = pred_loss + stop_loss
		total_loss = fragment_decoding_loss+ rxn_decoding_loss + beta * (kl_loss) 

		return total_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss










