import torch
import torch.nn as nn
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from initializers import *
from utils import *

class RNNCell(nn.Module):
    def __init__(self, params):
        """
        RNN cell
        Args:
        params: dictionary with model params
        """
        super(RNNCell, self).__init__()
        seed_everything(params["seed"])
        
        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])
        
        if params["task"] == "MDXOR":
            n_inp = 3
        elif params["task"] == "Cue":
            n_inp = 2
        elif params["task"] == "Cos":
            n_inp = 1

        self.n_rec = params["n_rec"]
        self.n_out = params["n_out"]
        
        # declare network parameters
        self.bias1 = nn.Parameter(torch.Tensor(self.n_rec))
        self.w_inp = nn.Parameter(torch.Tensor(n_inp, self.n_rec))
        self.w_rec1 = nn.Parameter(torch.Tensor(self.n_rec, self.n_rec))
        self.w_out = nn.Parameter(torch.Tensor(self.n_rec, self.n_out))
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        self.b_out = nn.Parameter(torch.Tensor(self.n_out))
        self.w_out_scale = nn.Parameter(torch.Tensor(1, 1))
        self.gtf = params["GTF"]

        # Set requires_grad based on params
        for param_name in ["w_out_scale", "w_inp_scale", "w_inp", "w_out", "w_rec1", "bias1"]:
            getattr(self, param_name).requires_grad = params.get(f"train_{param_name}", True)

        # time constants
        self.dt = params["dt"]
        self.tau = params["tau_lims"]
        self.B = params["B"]
        # initialize parameters
        with torch.no_grad():
            self.initialize_parameters(params)

    def initialize_parameters(self, params):
        init_method = params.get("init_method", "neuro")

        # Initialize w_rec
        if init_method == "neuro":
            w_rec1 = initialize_w_rec(params)
            self.w_rec1.copy_(torch.from_numpy(w_rec1))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # Initialize other parameters 
        w_inp = initialize_w_inp(params)
        self.w_inp.copy_(torch.from_numpy(w_inp))
        self.w_inp_scale.fill_(params["scale_w_inp"])
        self.w_out_scale.fill_(params["scale_w_out"])
        
        # Initialize w_out
        if params["1overN_out_scaling"]:
            self.w_out.normal_(std=1 / self.n_rec)
        else:
            self.w_out.normal_(std=1 / np.sqrt(self.n_rec))

        # Initialize bias terms to zero
        self.bias1.zero_()
        self.b_out.zero_()

    def forward(self, input, h, y, mask, i, _, pred=False):

        alpha = self.dt / self.tau[0]
        
        h = (1 - alpha) * h + alpha * (torch.matmul(self.nonlinearity(h), self.w_rec1.t())  + self.bias1 + input.matmul(self.w_inp))
        
        output = self.w_out_scale * self.out_nonlinearity(h).matmul(self.w_out) + self.b_out
        
        e = (y[:,i,:]-output[:,:])*mask[:,i,:]

        c = self.B[0]*mask[:,i,0].unsqueeze(1) if pred == False else 0
        
        ridge = 1e-7                       
        I_N   = torch.eye(self.w_out.shape[1], device=self.w_out.device)

        pinv = torch.linalg.inv(self.w_out.T @ self.w_out + ridge*I_N) @ self.w_out.T

        if pred:
            h = h 
        else:
            if self.gtf:
                rGTF = (y[:,i,:]@pinv)
                h = (1-c)*h + c*rGTF #+ self.perturb*pert
            else:
                h = h + c*(e@pinv).detach()
        
        return h, output, _
    
def set_nonlinearity(param):
    """utility returning activation function"""
    if param == "tanh":
        return torch.tanh
    elif param == "identity":
        return lambda x: x
    elif param == "logistic":
        return torch.sigmoid
    elif param == "relu":
        return nn.ReLU()
    elif param == "softplus":
        softplus_scale = 1  # Note that scale 1 is quite far from relu
        nonlinearity = (
            lambda x: torch.log(1.0 + torch.exp(softplus_scale * x)) / softplus_scale
        )
        return nonlinearity
    elif type(param) == str:
        print("Nonlinearity not yet implemented.")
        print("Continuing with identity")
        return lambda x: x
    else:
        return param

class hRNNCell(nn.Module):
    def __init__(self, params):
        """
        RFLO
        """
        super(hRNNCell, self).__init__()
        seed_everything(params["seed"])
        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])
        if params["task"] == "MDXOR":
            n_inp = 3
        elif params["task"] == "Cue":
            n_inp = 2
        elif params["task"] == "Cos":
            n_inp = 1
        self.n_rec = params["n_rec"]
        self.n_out = params["n_out"]
        # declare network parameters
        self.bias1 = nn.Parameter(torch.Tensor(self.n_rec))
        self.w_inp = nn.Parameter(torch.Tensor(n_inp, params["n_rec"]))
        self.w_rec1 = nn.Parameter(torch.Tensor(params["n_rec"], params["n_rec"]))
        self.w_out = nn.Parameter(torch.Tensor(params["n_rec"], params["n_out"]))
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        self.b_out = nn.Parameter(torch.Tensor(params["n_out"]))
        self.w_out_scale = nn.Parameter(torch.Tensor(1, params["n_out"]))
        self.gtf = params["GTF"]

        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False
        if not params["train_w_inp"]:
            self.w_inp.requires_grad = False
        if not params["train_w_out"]:
            self.w_out.requires_grad = False
        if not params["train_w_rec1"]:
            self.w_rec1.requires_grad = False

        # time constants
        self.dt = params["dt"]
        self.tau = params["tau_lims"]
        self.B = params["B"]

        # initialize parameters
        with torch.no_grad():
            w_inp = initialize_w_inp(params)
            self.w_inp = self.w_inp.copy_(torch.from_numpy(w_inp))
            self.r_homeo = self.r_homeo.fill_(10)
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])
            self.f_out_scale = self.f_out_scale.fill_(params["scale_w_out"])
            w_rec1 = initialize_w_rec(params)
            self.w_rec1 = self.w_rec1.copy_(torch.from_numpy(w_rec1))

            # deep versus shallow learning?
            if params["1overN_out_scaling"]:
                self.w_out = self.w_out.normal_(std=1 / params["n_rec"])
            else:
                self.w_out = self.w_out.normal_(std=1 / np.sqrt(params["n_rec"]))

            self.bias1.zero_()
            self.b_out.zero_()

        np.save('w_out_scale_init.npy', self.w_out_scale.detach().numpy())

    def forward(self, input, h, y, mask, i, e_p, pred=False):
        """
        Do a forward pass through one timestep
        """

        alpha = self.dt / self.tau[0]

        #this is telling auto-grad to not compute non-local gradients, hence RFLO
        v = torch.matmul(self.nonlinearity(h), torch.eye(self.w_rec1.size(0)).to(self.w_rec1.device)*self.w_rec1.t()) \
                    + torch.matmul(self.nonlinearity(h).detach(), (1-torch.eye(self.w_rec1.size(0))).to(self.w_rec1.device)*self.w_rec1.t()) \
                    + input.matmul(self.w_inp) \
                    + self.bias1
        h = (1 - alpha) * h + alpha * v   
        output = self.w_out_scale * self.out_nonlinearity(h).matmul(self.w_out) + self.b_out
        e = (y[:,i,:]-output[:,:])*mask[:,i,:] 
        
        c = self.B[0]*mask[:,i,0].unsqueeze(1) if pred == False else 0
        ridge = 1e-7                                
        I_N   = torch.eye(self.w_out.shape[1], device=self.w_out.device)
        pinv = torch.linalg.inv(self.w_out.T @ self.w_out + ridge*I_N) @ self.w_out.T

        if pred or i == 0:
            h = h
            #add if-else block to use TF 
        else:
            h = h + c*(e_p@pinv).detach()  #using e_{t-1} instead of e_t for causality -> more bio-plausible
        return h, output, e
    
def set_nonlinearity(param):
    """utility returning activation function"""
    if param == "tanh":
        return torch.tanh
    elif param == "identity":
        return lambda x: x
    elif param == "logistic":
        return torch.sigmoid
    elif param == "relu":
        return nn.ReLU()
    elif param == "softplus":
        softplus_scale = 1 
        nonlinearity = (
            lambda x: torch.log(1.0 + torch.exp(softplus_scale * x)) / softplus_scale
        )
        return nonlinearity
    elif type(param) == str:
        print("Nonlinearity not yet implemented.")
        print("Continuing with identity")
        return lambda x: x
    else:
        return param
