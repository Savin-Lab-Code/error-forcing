import torch
import os, sys
import numpy as np
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
current_directory = os.getcwd() + "/.." 
sys.path.append(current_directory) 

def organize_files(params, training_params, out_dir, diff, seed, fname, grad_norms, wout, fout, train_loss, val_loss, converged, dRec, dOut):
    experiments_folder = os.path.join(out_dir, 'experiments')
    os.makedirs(experiments_folder, exist_ok=True)

    dxor_folder = os.path.join(experiments_folder, 'DXOR-randomlearned')

    # Determine the cell type (BPTT or RFLO), or EF or GTF
    #cell_type = 'BPTT' if params["cell_type"] == "RNN" else 'RFLO'
    cell_type = 'GTF' if params["GTF"]==True else 'EF'

    experiment_folder = f"{cell_type}_D{diff}_A{params['B'][0]}"
    full_experiment_path = os.path.join(dxor_folder, experiment_folder)
    os.makedirs(full_experiment_path, exist_ok=True)

    # Save the files in the new folder
    np.save(os.path.join(full_experiment_path, f'wout_{converged}_{fname}_{seed}.npy'), wout)
    np.save(os.path.join(full_experiment_path, f'gn_{converged}_{fname}_{seed}.npy'), grad_norms)
    np.save(os.path.join(full_experiment_path, f'train_{converged}_{fname}_{seed}.npy'), train_loss)
    np.save(os.path.join(full_experiment_path, f'val_{converged}_{fname}_{seed}.npy'), val_loss)

    print(f"Files saved in: {full_experiment_path}")
    return full_experiment_path


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
