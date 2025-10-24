from turtle import clearstamps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
import time
import os
import sys
import math
from tasks.seqDS import *
from utils import *

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from model import *


def MSE_loss(output, target, mask):
    loss = torch.sum((output - target)**2 * mask) / torch.sum(mask)
    return loss

def train_rnn(
    rnn,
    training_params,
    task,
    sync_wandb=False,
    wandb_log_freq=10,
    x0=None,
    params=None,
    task_params=None,
    out_dir=None,
    fname=None,
    max_retries = 20, 
):
    """
    Train a RNN
    """
    seed_everything(params["seed"])
    # do a train / validation split
    def initialize_dataloaders():
        if training_params["validation_split"]:
            if params.get("task") == "MDXOR" or params.get("task") == "Cos" or params.get("task") == "Cue":
                indices = list(range(task.len))
                split = int(np.floor(training_params["validation_split"] * task.len))
                np.random.shuffle(indices)
                train_indices, val_indices = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)
                train_loader = DataLoader(
                    task, batch_size=training_params["batch_size"], sampler=train_sampler
                )
                valid_loader = DataLoader(
                    task, batch_size=training_params["batch_size"], sampler=valid_sampler
                )
                training_params["val_indices"] = val_indices
                print(str(len(train_indices)) + " trials in train set")
                print(str(len(val_indices)) + " trials in validation set")
        else:
            train_loader = DataLoader(
                task, batch_size=training_params["batch_size"], shuffle=True
            )
            valid_loader = None

        return train_loader, valid_loader
    
    dataloader, valid_dataloader = initialize_dataloaders()

    # cuda management
    if training_params["cuda"]:
        if not torch.cuda.is_available():
            #print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(training_params['local_rank'])
            device = torch.device(f"cuda:{training_params['local_rank']}")
    else:
        device = torch.device("cpu")
    
    rnn.to(device = device)


    reg_fns = []
    reg_costs = []
    # regularisation
    # l2 reg on firing rates 
    if training_params["l2_rates_cost"]:
        reg_fns.append(l2_rates_loss)
        reg_costs.append(training_params["l2_rates_cost"])

    # svd condition penalty
    if training_params.get("svd_ratio_cost", 0) > 0:
        reg_fns.append(svd_ratio_loss)
        reg_costs.append(training_params["svd_ratio_cost"])

    if len(reg_fns) == 0:
        reg_fns.append(zero_loss)
        reg_costs.append(0)


    reg_costs = torch.tensor(reg_costs, device=device)
    #print('reg coststs', reg_costs)

    # optimizer
    if training_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(rnn.parameters(), lr=training_params["lr"])
    if training_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(rnn.parameters(), lr=training_params["lr"])

    gamma = np.exp(
        np.log(training_params["lr_end"] / training_params["lr"])
        / training_params["n_epochs"]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma, last_epoch=-1
    )

    retry_count = 0
    while retry_count < max_retries:
        # start timer before training
        time0 = time.time()

        # set rnn to training mode
        rnn.train()

        losses = []
        val_losses = []
        reg_losses = []
        grad_norms = [] 

        #variables for convergence check
        convergence_threshold = 0.1 if params['task'] == "MDXOR" else 0.01
        convergence_threshold = 0.1 if params['task'] == "Cue" else 0.01
        convergence_count = 0
        converged_epoch = None

        # start training loop
        wout_save = []
        fout_save = []
        for i in range(training_params["n_epochs"]):
            loss_ep = 0.0
            reg_loss_ep = torch.zeros(len(reg_fns), device=device)
            num_len = 0
            vall_ep = 0.0
            val_num_len = 0
            # loop through dataloader
            for x, y, m in dataloader:
                x = x.to(device=device)
                y = y.to(device=device)
                m = m.to(device=device)

                _, rates, y_pred, _ = rnn(x, y, m, pred=False, x0=x0)
                # Store gradients of w_rec
                wout_save.append(rnn.cell.w_out.detach().cpu().numpy()) 

                optimizer.zero_grad()
                task_loss = MSE_loss(y_pred, y, m)
                #task_loss = task_loss.sum() / m.sum()
                reg_loss = torch.stack(
                    [
                        reg_fn(rates[:, 1:], rnn=rnn.cell, mask=m, stim=x)
                        for reg_fn in reg_fns
                    ]
                ).squeeze()  # , device=device)

                # grad descent
                loss = task_loss + torch.sum(reg_loss * reg_costs) #+ temp
                loss.backward()

                total_grad_norm = 0.0
                for param in rnn.parameters():
                    if param.grad is not None:
                        param_grad_norm = torch.norm(param.grad, p=2)
                        total_grad_norm += param_grad_norm.item()
                grad_norms.append(total_grad_norm)
                
                # clip weights to avoid explosion of gradients
                if training_params["clip_gradient"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        rnn.parameters(), training_params["clip_gradient"]
                    )

                # update weights
                optimizer.step()

                num_len += 1
                loss_ep += task_loss.item()
                reg_loss_ep += reg_loss*reg_costs

            loss_ep = loss_ep / num_len
            reg_loss_ep /= num_len
            reg_loss_ep = reg_loss_ep.tolist()

            # potentially calculate validation loss, don't do backprop here
            if training_params["validation_split"]:
                # loop through dataloader
                with torch.no_grad():
                    for x_val, y_val, m_val in valid_dataloader:
                        x_val = x_val.to(device=device)
                        y_val = y_val.to(device=device)
                        m_val = m_val.to(device=device)
                        shp1 = x_val.shape[0]
                        shp2 = x_val.shape[1]

                        y_zero = torch.zeros((shp1, shp2, 1)).to(device=device)
                        _, _, y_pred_val, _ = rnn(x_val, y_zero, m_val, pred=True, x0=x0)

                        val_num_len += 1
                        val_loss = MSE_loss(y_pred_val, y_val, m_val)  
                        vall_ep += val_loss
                    vall_ep /= val_num_len

                if i % 10 == 0:
                    print(
                        "epoch {:d} / {:d}: time={:.1f} s, task loss={:.5f}, val loss = {:.5f}, reg loss=".format(
                            i + 1,
                            training_params["n_epochs"],
                            time.time() - time0,
                            loss_ep,
                            vall_ep,
                        )
                        + str(["{:.5f}"] * len(reg_loss_ep))
                        .format(*reg_loss_ep)
                        .strip("[]")
                        .replace("'", ""),
                    )
            else:
                print(
                    "epoch {:d} / {:d}: time={:.1f} s, task loss={:.15f}, reg loss=".format(
                        i + 1, training_params["n_epochs"], time.time() - time0, loss_ep
                    )
                    + str(["{:.5f}"] * len(reg_loss_ep))
                    .format(0)
                    .strip("[]")
                    .replace("'", ""),
                    # end="\r",
                )

            if math.isinf(loss_ep) or math.isnan(loss_ep) or loss_ep > 10000:
                print(f"Loss exceeded threshold at epoch {i + 1}, retrying training...")
                retry_count += 1
                params["seed"] = params["seed"] + np.random.randint(0, 1000)
                seed_everything(params["seed"])
                rnn = RNN(params)  # reinitialize the RNN
                rnn.to(device=device)
                optimizer = torch.optim.Adam(rnn.parameters(), lr=training_params["lr"])  # reinitialize the optimizer
                dataloader, valid_loader = initialize_dataloaders()  # reinitialize data loaders
                if retry_count >= max_retries:
                    print(f"Training failed after {max_retries} retries. Assessing high loss...")
                    losses.append(10.0)
                break
            else:
                losses.append(loss_ep)
                retry_count = max_retries

            if training_params["validation_split"] == False:
                vall_ep = 0
            if isinstance(vall_ep, torch.Tensor):
                vall_ep = vall_ep.cpu().numpy() 
            val_losses.append(vall_ep)
            reg_losses.append(reg_loss_ep) 
            scheduler.step()

            # check for convergence
            if vall_ep < convergence_threshold:
                convergence_count += 1
                if convergence_count == 10:
                    converged_epoch = i + 1
                    print(f"\nConvergence achieved at epoch {converged_epoch}")
                    break
            else:
                convergence_count = 0

    # if the loop completed without convergence
    if converged_epoch is None:
        print("\nTraining completed without achieving convergence")
    else:
        print(f"Saving model at convergence (epoch {converged_epoch})")
        
    print("\nDone. Training took %.1f sec." % (time.time() - time0))
    fname = fname + "dur" + str(time.time() - time0)
    print("Saving model to " + os.path.join(out_dir, fname))
    training_params["val_loss"] = val_losses
    training_params["train_loss"] = losses

    # save trained network
    save_rnn(os.path.join(out_dir, fname), rnn, params, task_params, training_params)

    rnn.eval()
    
    return losses, val_losses, fname, wout_save, fout_save, grad_norms, converged_epoch, _, _


def load_rnn(name, device="cpu"):
    """
    loads an RNN
    """

    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)
    with open(task_params_file, "rb") as f:
        task_params = pickle.load(f)
    with open(training_params_file, "rb") as f:
        training_params = pickle.load(f)
        
    print(params)
    model = RNN(params)
    model.load_state_dict(
        torch.load(state_dict_file, map_location=torch.device(device))
    )

    return model, params, task_params, training_params


def save_rnn(name, model, params, task_params, training_params):
    """
    saves an RNN
    """
    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"
    with open(params_file, "wb") as f:
        pickle.dump(params, f)
    with open(training_params_file, "wb") as f:
        pickle.dump(training_params, f)
    with open(task_params_file, "wb") as f:
        pickle.dump(task_params, f)

    torch.save(model.state_dict(), state_dict_file)


def l2_rates_loss(rates, **kwargs):
    return torch.mean(rates**2)

def zero_loss(x, **kwargs):
    return torch.zeros(1, device=x.device)

def svd_ratio_loss(rates, *, rnn, eps: float = 1e-8, **kwargs):
    """
    Penalises a poor condition number of the read-out matrix
    """
    W = rnn.w_out            
    if W.shape[1] == 1:
        w = rnn.w_out.squeeze()              
        ratio = torch.linalg.norm(w) + eps     
    else:
        s = torch.linalg.svdvals(W)  # singular values, descending
        if s.numel() < 2:            # degenerate case
            return torch.zeros(1, device=W.device)
        ratio = s[0] / (s[-1] + eps)    
    return (1.0 - ratio).pow(2)      
