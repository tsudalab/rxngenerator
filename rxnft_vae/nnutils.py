import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def create_var(tensor, requires_grad = None):
	if requires_grad is None:
		return Variable(tensor)#.cuda()
	else:
		return Variable(tensor, requires_grad=requires_grad)#.cuda()
def index_select_ND(source, dim, index):
	index_size = index.size()
	suffix_dim = source.size()[1:]
	final_size = index_size + suffix_dim
	target = source.index_select(dim, index.view(-1))
	return target.view(final_size)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x,sum_h], dim=1)
    z = nn.Sigmoid()(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = nn.Sigmoid()(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = nn.Tanh()(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h
def attention(encoder_outputs, hiddens):
    # encoder_output: B x n x d
    # hidden :        B x d
    #return torch.zeros_like(hiddens)
    encoder_hidden_outs = torch.stack(encoder_outputs, dim=0)
    hiddens = hiddens.unsqueeze(1) # hidden: B x 1 x d
    t_hiddens = torch.transpose(hiddens, 1, 2) # hidden: B x d x 1
    s = torch.bmm(encoder_hidden_outs, t_hiddens) # B x n x 1
    attention_weight = nn.Softmax(dim=1)(s)
    unsq_weight = attention_weight[:,:, 0].unsqueeze(2)
    weighted_outputs = encoder_hidden_outs * unsq_weight
    weighted_sum = torch.sum(weighted_outputs, axis=1)
    #print(weighted_sum.size())

    return weighted_sum
