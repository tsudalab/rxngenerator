import torch
import torch.nn as nn
from nnutils import create_var, GRU
from fragment import FragmentVocab, FragmentTree, FragmentNode
from chemutils import set_atommap, enum_assemble, enum_attach
import copy


MAX_NB = 16
MAX_DECODING_LEN = 100

class FTDecoder(nn.Module):
	def __init__(self, ftvocab, hidden_size, latent_size, embedding=None):
		super(FTDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.ftvocab = ftvocab
		self.ftvocab_size = ftvocab.size()

		if embedding is None:
			self.embedding = nn.Embedding(self.ftvocab_size, hidden_size)
		else:
			self.embedding = embedding
		# GRU weights
		self.W_z = nn.Linear(2 * hidden_size, hidden_size)
		self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
		self.W_r = nn.Linear(hidden_size, hidden_size)
		self.W_h = nn.Linear(2 * hidden_size, hidden_size)

		self.W = nn.Linear(latent_size + hidden_size, hidden_size)
		self.U = nn.Linear(latent_size + 2 *hidden_size, hidden_size)

		self.W_o = nn.Linear(hidden_size, self.ftvocab_size)
		self.U_s = nn.Linear(hidden_size, 1)

		self.pred_loss = nn.CrossEntropyLoss(size_average=False)
		self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)
		#self.dropout = nn.Dropout(self.dropout_p)

	def get_trace(self, node):
		super_root = FragmentNode("")
		super_root.idx = -1
		trace =[]
		dfs(trace, node, super_root)
		return [(x.smiles, y.smiles, z) for x, y, z in trace]

	def decode(self, tree_vec, prob_decode=True):
		stack, trace =[],[]
		init_hidden = create_var(torch.zeros(1, self.hidden_size))
		zero_pad = create_var(torch.zeros(1,1,self.hidden_size))

		# root prediction
		root_hidden = torch.cat([init_hidden, tree_vec], dim=1)

		root_hidden = nn.ReLU()(self.W(root_hidden))
		root_score = self.W_o(root_hidden)
		_, root_wid = torch.max(root_score, dim=1)
		root_wid = root_wid.item()
		id = 0
		nodes={}
		edges=[]
		nodes[id] = self.ftvocab.get_smiles(root_wid)
		id+=1

		root = FragmentNode(self.ftvocab.get_smiles(root_wid))
		root.wid = root_wid
		root.idx = 0
		stack.append((root, self.ftvocab.get_slots(root.wid)))


		all_nodes = [root]
		h={}
		for step in range(MAX_DECODING_LEN):
			node_x, fa_slot = stack[-1]
			cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
			if len(cur_h_nei) > 0:
				cur_h_nei=torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
			else:
				cur_h_nei = zero_pad
			cur_x = create_var(torch.LongTensor([node_x.wid]))
			cur_x = self.embedding(cur_x)

			# predict stop
			cur_h = cur_h_nei.sum(dim=1)
			stop_hidden = torch.cat([cur_x, cur_h, tree_vec], dim=1)
			stop_hidden = nn.ReLU()(self.U(stop_hidden))
			stop_score = nn.Sigmoid()(self.U_s(stop_hidden)*20).squeeze()

			if prob_decode:
				#print(1.0 - stop_score)
				backtrack = (torch.bernoulli(1.0 - stop_score)==1)
			else:
				backtrack = (stop_score.item() < 0.5)
			if not backtrack:
				# predict next clique
				new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
				pred_hidden = torch.cat([new_h, tree_vec], dim=1)
				pred_hidden = nn.ReLU()(self.W(pred_hidden))
				#print(self.W_o(pred_hidden)*20)
				pred_score = nn.Softmax()(self.W_o(pred_hidden)*20)
				#print(pred_score)
				if prob_decode:
					sort_wid = torch.multinomial(pred_score, 5)
					sort_wid=sort_wid[0,:]
					sort_wid = sort_wid.numpy()
				else:
					_, sort_wid = torch.sort(pred_score, dim=1, descending=True)
					sort_wid = sort_wid[0,:].numpy()
				next_wid = None
				for wid in sort_wid[:5]:
					slots = self.ftvocab.get_slots(wid)
					node_y = FragmentNode(self.ftvocab.get_smiles(wid))
					if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
						next_wid = wid
						next_slots = slots
						break
				if next_wid is None:
					backtrack = True
				else:
					node_y = FragmentNode(self.ftvocab.get_smiles(next_wid))
					node_y.wid = next_wid
					node_y.idx = len(all_nodes)
					node_y.neighbors.append(node_x)
					h[(node_x.idx, node_y.idx)] = new_h[0]
					stack.append((node_y, next_slots))
					all_nodes.append(node_y)
					#tree.nodes.append(node_y)
					nodes[id] = self.ftvocab.get_smiles(next_wid)
					id+=1
					edges.append((node_x.idx, node_y.idx))
					edges.append((node_y.idx, node_x.idx))
			if backtrack:
				if len(stack) == 1:
					break
				node_fa, _ = stack[-2]
				cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
				if len(cur_h_nei) > 0:
					cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
				else:
					cur_h_nei = zero_pad
				new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
				h[(node_x.idx, node_fa.idx)] = new_h[0]
				node_fa.neighbors.append(node_x)
				stack.pop()
		tree = FragmentTree(smiles=None)
		for id, node in nodes.items():
			n = FragmentNode(node)
			n.wid = self.ftvocab.get_index(node)
			n.idx = id
			tree.nodes.append(n)
		for edge in edges:
			idx, idy = edge[0], edge[1]
			tree.nodes[idx].add_neighbor(tree.nodes[idy])					
		return tree








	def forward(self, tree_batch, tree_vecs):
		super_root = FragmentNode("")
		super_root.idx = -1

		stop_hiddens, stop_targets =[],[]
		pred_hiddens, pred_targets, pred_tree_vecs =[],[], []

		traces =[]
		for tree in tree_batch:
			s = []
			dfs(s, tree.nodes[0], super_root)
			traces.append(s)
			for node in tree.nodes:
				node.neighbors=[]

		# predict root
		pred_tree_vecs.append(tree_vecs)
		pred_targets.extend([tree.nodes[0].wid for tree in tree_batch])
		pred_hiddens.append(create_var(torch.zeros(len(tree_batch), self.hidden_size)))

		max_iter = max([len(tr) for tr in traces])
		padding = create_var(torch.zeros(self.hidden_size), False)
		h={}

		for t in range(max_iter):
			prop_list=[]
			batch_list=[]
			for i, plist in enumerate(traces):
				if len(plist) > t:
					prop_list.append(plist[t])
					batch_list.append(i)
			cur_x =[]
			cur_h_nei, cur_o_nei =[],[]
			for node_x, real_y, _ in prop_list:
				cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
				pad_len = MAX_NB - len(cur_nei)
				cur_h_nei.extend(cur_nei)
				cur_h_nei.extend([padding] * pad_len)

				cur_nei = [h[node_y.idx, node_x.idx] for node_y in node_x.neighbors]
				pad_len = MAX_NB - len(cur_nei)
				cur_o_nei.extend(cur_nei)
				cur_o_nei.extend([padding] * pad_len)

				cur_x.append(node_x.wid)
			cur_x = create_var(torch.LongTensor(cur_x))
			cur_x = self.embedding(cur_x)

			cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
			new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

			cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
			cur_o = cur_o_nei.sum(dim=1)


			stop_target=[]
			pred_target, pred_list =[],[]
			for i,m in enumerate(prop_list):
				node_x, node_y, direction = m
				x,y = node_x.idx, node_y.idx
				h[(x,y)] = new_h[i]
				node_y.neighbors.append(node_x)
				stop_target.append(direction)
				if direction==1:
					pred_target.append(node_y.wid)
					pred_list.append(i)
			# hidden states for stop prediction
			cur_batch = create_var(torch.LongTensor(batch_list))
			cur_tree_vec = tree_vecs.index_select(0, cur_batch)
			stop_hidden = torch.cat([cur_x, cur_o, cur_tree_vec], dim=1)
			stop_hiddens.append(stop_hidden)
			stop_targets.extend(stop_target)

			# hidden states for clique prediction
			if len(pred_list) > 0:
				batch_list = [batch_list[i] for i in pred_list]
				cur_batch = create_var(torch.LongTensor(batch_list))
				pred_tree_vecs.append(tree_vecs.index_select(0, cur_batch))

				cur_pred = create_var(torch.LongTensor(pred_list))
				pred_hiddens.append(new_h.index_select(0, cur_pred))
				pred_targets.extend(pred_target)
		# last stop at root
		cur_x, cur_o_nei =[],[]
		for tree in tree_batch:
			node_x = tree.nodes[0]
			cur_x.append(node_x.wid)
			cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
			pad_len = MAX_NB - len(cur_nei)
			cur_o_nei.extend(cur_nei)
			cur_o_nei.extend([padding] * pad_len)
		cur_x = create_var(torch.LongTensor(cur_x))
		cur_x = self.embedding(cur_x)
		cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
		cur_o = cur_o_nei.sum(dim=1)

		stop_hidden = torch.cat([cur_x, cur_o, tree_vecs], dim=1)
		stop_hiddens.append(stop_hidden)
		stop_targets.extend([0]*len(tree_batch))

		# predict next clique
		pred_hiddens = torch.cat(pred_hiddens, dim=0)
		pred_tree_vecs = torch.cat(pred_tree_vecs, dim=0)
		pred_vecs = torch.cat([pred_hiddens, pred_tree_vecs], dim=1)
		pred_vecs = nn.ReLU()(self.W(pred_vecs))
		pred_scores = self.W_o(pred_vecs)
		pred_targets = create_var(torch.LongTensor(pred_targets))

		pred_loss = self.pred_loss(pred_scores, pred_targets)/len(tree_batch)
		_, preds = torch.max(pred_scores, dim=1)
		pred_acc = torch.eq(preds, pred_targets).float()
		pred_acc = torch.sum(pred_acc)/pred_targets.nelement()

		# predict stop
		stop_hiddens = torch.cat(stop_hiddens, dim=0)
		stop_vecs = nn.ReLU()(self.U(stop_hiddens))
		stop_scores = self.U_s(stop_vecs).squeeze()
		stop_targets=create_var(torch.FloatTensor(stop_targets))

		stop_loss = self.stop_loss(stop_scores, stop_targets)/len(tree_batch)
		stops = torch.ge(stop_scores,0).float()
		stop_acc = torch.eq(stops, stop_targets).float()
		stop_acc = torch.sum(stop_acc)/stop_targets.nelement()

		return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

def dfs(stack, x, fa):
	for y in x.neighbors:
		if y.idx == fa.idx:
			continue
		stack.append((x, y, 1))
		dfs(stack, y, x)
		stack.append((y, x, 0))

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0# and sum(aroma_scores) >= 0


