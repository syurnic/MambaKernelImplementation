#Tinystories.

import math
from typing import List
import datasets, torch, einops, bitsandbytes
import torch.nn as nn
import torch.nn.functional as F

#hyperparamters
d_model : int = 128
n_layers : int = 4
d_state : int= 16
d_conv : int = 4
vocab_size : int = 5000

expension_factor : int = 2
class MambaConfig:
    def __init__(self, d_model : int, n_layers : int, d_state : int, d_conv : int):
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.vocab_size = vocab_size

class MambaBlockConfig:
    def __init__(self, config : MambaConfig):
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv

class MambaBlock(nn.Module):
    #Should handle dimension expension and shrinkage.
    #Should add gating.
    def __init__(self, config : MambaBlockConfig):
        super(MambaBlock, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state #hidden space dimension
        self.d_conv = config.d_conv
        self.d_inner = self.d_model * expension_factor
        self.expand_mat = nn.Linear(self.d_model, 2 * self.d_inner)
        self.shrink_mat = nn.Linear(self.d_inner, self.d_model)
        self.deltaaddparam = torch.nn.Parameter(torch.randn(self.d_model))

        self.convolution = nn.Conv1d(self.d_inner,self.d_inner, kernel_size = self.d_conv)
        self.A = nn.Linear(self.d_inner, self.d_state)
        self.B_projector = nn.Linear(self.d_model, self.d_state)
        self.C_projector = nn.Linear(self.d_model, self.d_state)
        self.delta_projector = nn.Linear(self.d_model, self.d_model)

        self.hidden = torch.zeros(self.d_state)
    def discretize(self, delta, A, B):
        #Assume zero order hold
        A_bar = torch.exp(delta @ A)
        B_bar = (1 / (delta @ A)) * (torch.exp(delta @ A) - 1) * (delta * B)
        return A_bar, B_bar

    def one_token_forward(self, A_bar, B_bar, x_t):
        self.hidden = A_bar @ self.hidden + B_bar @ x_t

    def mambaOp(self, A_prev, B_prev, A_next, B_next):
        #A_i includes only matrix A. hidden vector is not included
        #B_i include input vector x. (i.e. B_i = B_bar @ x_t)
        return A_next @ B_next, A_next @ B_prev + B_next

    def parallelScan(self, A_bar, B_bar, C, x):
        B_producted = B_bar @ x
        scan_result = 0

        num_steps = math.ceil(torch.log2(B_bar.shape[1]))
        for i in range(num_steps):

        return C @ scan_result
    def SSMforward(self, x, parallel : bool):
        #x : (Batch, Length, D_model)
        #First implement when parallel scan is available
        x = F.pad(x, (0, 0, self.d_conv - 1, 0))
        x = self.convolution(x.permute(0, 2, 1)).permute(0, 2, 1)
        B = self.B_projector(x)
        C = self.C_projector(x)
        delta = F.softplus(self.deltaaddparam + self.delta_projector(x))
        A_bar, B_bar = self.discretize(delta, self.A, B)
        if parallel:
            self.parallelScan(A_bar, B_bar, C, x)


    def forward(self, x, parallel : bool):
        x = self.expand_mat(x)
        SSM_x, gate_x = torch.split(x, [self.d_inner, self.d_inner], dim=-1)
        #SSM calculation should be done
        SSM_x = self.SSMforward(x, parallel)
        gate_x = F.silu(gate_x) #gating
        out_x = torch.mul(gate_x, SSM_x)
        out_x = self.shrink_mat(out_x)

class Mamba:
    def __init__(self, config : MambaConfig):
        super(Mamba, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.mambaBlockConfig : MambaBlockConfig = MambaBlockConfig(config)
        self.vocab_layer = nn.Linear(self.config.d_model, self.config.vocab_size)

        self.MambaBlocks : List[MambaBlock] = [MambaBlock(self.mambaBlockConfig) for _ in range(config.n_layers)]


    def forward(self, x, parallel : bool = True):
        #x is well tokenized
        x = self.embedding(x)
        for i in range(self.config.n_layers):
            x = self.MambaBlocks[i](x, parallel=parallel)
        x = self.vocab_layer(x)
        return torch.argmax(x, dim=-1)