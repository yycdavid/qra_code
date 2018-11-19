import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

torch.manual_seed(10)

class ATTENTION(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(ATTENTION, ATTENTION).add_config(cfgparser)
        cfgparser.add_argument("--n_d", "--d", type=int, help="embedding dimension")


    def __init__(self, embedding_layer, configs):
        super(ATTENTION, self).__init__(configs)
        self.embedding_layer = embedding_layer
        self.embedding_layer.embedding.weight.requires_grad = True
        self.embedding = self.embedding_layer.embedding
        self.n_e = embedding_layer.n_d
        self.n_d = self.n_e
        self.use_cuda = configs.cuda
        self.s = nn.Linear(self.n_e, 1, bias=False)
        self.weight_init()
        self.n_out = self.n_d
        self.build_output_op()

    def weight_init(self):
        nn.init.xavier_uniform(self.s.weight)

    def forward(self, batch):
        # batch is of size (len, batch_size)
        emb = self.embedding(batch)  # (len, batch_size, n_e)
        emb = Variable(emb.data)
        assert emb.dim() == 3

        emb_resized = emb.view(-1, self.n_e)
        s_out_resized = self.s(emb_resized) # (len, batch_size, 1)
        s_out = s_out_resized.view(emb.size()[0], emb.size()[1])

        # get mask
        padid = self.embedding_layer.padid
        mask = (batch != padid).type(torch.ByteTensor) # (len, batch_size)
        if self.use_cuda:
            mask = mask.cuda()
        # Compute sequence embedding
        s_out_t = torch.transpose(s_out, 0, 1).squeeze(2) # (batch_size, len)
        mask_t = torch.transpose(mask, 0, 1) # (batch_size, len)
        mask_n = Variable(1-mask_t.data)
        a = F.softmax(s_out_t.masked_fill_(mask_n, float('-inf'))) # (batch_size, len)
        output = torch.bmm(a.unsqueeze(1), torch.transpose(emb,0,1)).squeeze(1) # (batch_size, n_out)

        return output