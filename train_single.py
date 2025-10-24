"""
Train one model run

The entire repo is mostly adapted from the original codebases at https://github.com/mackelab/phase-limit-cycle-RNNs, and https://github.com/mackelab/smc_rnns
"""
import argparse, os, sys, time, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
current_directory = os.getcwd() + "/.." 
sys.path.append(current_directory) 
from model import *
from train import *
from tasks.seqDS import *
import time
from utils import *
import pickle

def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--B",            type=float,  default=0.0)
    p.add_argument("--batch_size",   type=int,    default=128)
    p.add_argument("--cell",         type=str,    default="RNN")
    p.add_argument("--diff",         type=float,  default=-55)
    p.add_argument("--g",            type=float,  default=1.5)
    p.add_argument("--gtf",          action="store_true")
    p.add_argument("--init_method",  type=str,    default="neuro")
    p.add_argument("--lr",           type=float,  default=1e-3)
    p.add_argument("--perturb",      type=float,  default=0.0)
    p.add_argument("--seed",         type=int,    default=0)
    p.add_argument("--task",         type=str,    default="Cos", choices=["Cos", "MDXOR", 'Cue'])
    p.add_argument("--cuda",         action="store_true", help="Force CUDA if available")
    p.add_argument("--epochs",       type=int, default=200)
    return p


def make_param_dict(args) -> tuple[dict, dict]:

    params = {
        # model / task ---------------------------------------------------
        "task": args.task,
        "cell_type": args.cell,
        "B": [args.B],
        "seed": args.seed,
        "nonlinearity": "tanh",
        "out_nonlinearity": "identity",
        "p_inp": 1,
        "n_rec": 50,
        "p_rec": 1,
        "n_out": 1,
        "scale_w_inp": 1,
        "scale_w_out": 1,
        "scale_f_out": 1,
        "1overN_out_scaling": True,
        "train_w_inp": False,
        "train_w_rec1": True,
        "w_rec_dist": "gauss",
        "spectr_rad": args.g,
        "train_w_inp_scale": False,
        "train_w_out": True,
        "train_w_out2": True,
        "train_w_out_scale": False,
        "train_f_out": True,
        "train_f_out_scale": False,
        "train_b_out": False,
        "train_r_homeo": False,
        "tau_lims": [10],
        "dt": 1,
        "scale_x0": 1,
        "GTF": args.gtf,
        "train_bias1": True,
        "init_method": args.init_method,
        "perturb": args.perturb,
    }

    training_params = {
        "n_epochs": args.epochs,
        "lr": args.lr,
        "lr_end": args.lr,
        "batch_size": args.batch_size,
        "clip_gradient": None,
        "cuda": args.cuda and torch.cuda.is_available(),
        "loss_fn": "mse",
        "optimizer": "adam",
        "l2_rates_cost": 0.00001,
        "l2_cov_cost": 0,
        "validation_split": 0.1,
        "local_rank": 0,    
        "svd_ratio_cost": 1e-6,# keeps your original train_rnn signature happy
    }
    return params, training_params


def single(params, training_params, diff):
    
    seed_everything(params["seed"])

    # task ---------------------------------------------------------------
    task_params = {"dt": 1, "delay": 0.01, "delay_resp": diff}

    if params['task'] == 'MDXOR':
        ds = MDXOR(task_params)

    elif params['task'] == 'Cos':
        ds = CosineContextDataset(task_params, dataset_len=5000)

    elif params['task'] == 'Cue':
        ds = Cue(task_params, dataset_len=216)
    # model --------------------------------------------------------------
    rnn = RNN(params)
    rnn.train()

    # training -----------------------------------------------------------
    out_dir = os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    print('starting', flush=True)
    fname = time.strftime("%m%d-%H%M%S")  # simple timestamp
    train_loss, val_loss, fname, wout_save, fout_save, grad_norms, converged, _, _ = train_rnn(
        rnn,
        training_params,
        ds,
        sync_wandb=False,      # <‑‑ not using wandb any more
        params=params,
        task_params=task_params,
        out_dir=out_dir,
        fname=fname,
    )

    rnn.eval()
    device_test = torch.device("cuda" if training_params["cuda"] else "cpu")
    rnn_test, _, _, _ = load_rnn(os.path.join(out_dir, fname), device_test)

    dl = DataLoader(ds, batch_size=200, shuffle=True)
    inp, tgt, msk = next(iter(dl))
    _, _, _, _, test_loss = predict(rnn_test, inp, tgt, msk, pred=True)
    print(f"test_loss: {test_loss:.6f}")

    # bookkeeping --------------------------------------------------------
    results_dir = os.path.join(ROOT, "runs")
    os.makedirs(results_dir, exist_ok=True)
    organize_files(
        params,
        training_params,
        results_dir,
        diff,
        params["seed"],
        fname,
        grad_norms,
        wout_save,
        fout_save,
        train_loss,
        val_loss,
        converged,
        None, 
        None,
    )
    return test_loss

def main():
    args = get_arg_parser().parse_args()
    params, training_params = make_param_dict(args)
    _ = single(params, training_params, args.diff)

if __name__ == "__main__":
    main()