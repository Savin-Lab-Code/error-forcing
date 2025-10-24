import numpy as np
import torch 
from utils import *

def initialize_w_rec(params):
    """
    Initializes recurrent weight matrix
    """
    seed_everything(params["seed"])
    w_rec = np.zeros((params["n_rec"], params["n_rec"]), dtype=np.float32)
    rec_idx = np.where(
        np.random.rand(params["n_rec"], params["n_rec"]) < params["p_rec"]
    )

    if params["w_rec_dist"] == "gauss":
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )
    else:
        print("WARNING: initialization not implemented, use Gauss")
        print("continuing with Gauss")
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )

    return w_rec


def initialize_w_inp(params):
    """
    Initializes input weight matrix
    """
    seed_everything(params["seed"])
    if params["task"] == "MDXOR":
        n_inp = 3
    elif params["task"] == "Cue":
        n_inp = 2
    elif params["task"] == "Cos":
        n_inp = 1
    w_task = np.zeros((params["n_rec"], n_inp), dtype=np.float32)
    idx = np.array(
        np.where(np.random.rand(params["n_rec"], n_inp) < params["p_inp"])
    )
    w_task[idx[0], idx[1]] = np.random.randn(len(idx[0])) * np.sqrt(1 / params["p_inp"])

    return w_task.T