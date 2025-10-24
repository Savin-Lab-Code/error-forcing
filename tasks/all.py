import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence
import random, math

class MDXOR(Dataset):
    def __init__(self,task_params,dataset_len=512):
        self.task_params = task_params
        self.dt_sec = self.task_params["dt"]/1000
        self.delay_resp = int(self.task_params["delay_resp"]/self.dt_sec)
        self.delay = int(0.01/self.dt_sec) #rflo 0.01, bptt 0.05
        self.n_inp = 3
        self.n_out = 1
        
        self.inp_dur = self.delay 
        self.resp_dur = self.delay
        self.f_dly_dur = self.delay
        self.trial_len = (self.resp_dur + 2*self.inp_dur + self.delay + self.delay//2 + self.delay_resp)
        self.len = dataset_len

    def __getitem__(self, idx):
        return self.create_stim()
    
    def __len__(self):
        return self.len
    
    def create_stim(self):
        noise = torch.randint(-self.delay//2, self.delay//2, (1,)).item()
        delay_resp2 = self.delay_resp + noise
        inp = torch.zeros((self.trial_len, self.n_inp), dtype =torch.float32) 
        options = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        random_index = torch.randint(0, 4, (1,)).item()
        stim = options[random_index]
        
        inp[:self.inp_dur, 0] = stim[0]
        inp[self.inp_dur + self.delay : 2*self.inp_dur + self.delay , 1] = stim[1]
        inp[:, 2] = 1
        inp[2*self.inp_dur + self.delay + delay_resp2 : self.resp_dur + 2*self.inp_dur + self.delay + delay_resp2, 2] = 0

        target = torch.zeros((self.trial_len, 1), dtype =torch.float32) 
        ans = (stim[0] ^ stim[1]) + 1
        target[2*self.inp_dur + self.delay + delay_resp2 : self.resp_dur + 2*self.inp_dur + self.delay + delay_resp2] = ans 

        mask = torch.zeros((self.trial_len, 1), dtype =torch.float32)
        mask[2*self.inp_dur + self.delay + delay_resp2 : self.resp_dur + 2*self.inp_dur + self.delay + delay_resp2, 0] = 1
        return inp, target, mask
    
       
class Cue(Dataset):

    def __init__(
            self,
            task_params: dict,
            dataset_len: int = 1024,
            post_delay: Optional[int] = None   
        ):
        super().__init__()
        self.p = task_params
        self.dt = 1.0 / 1000.0

        self.cue_dur     = int(0.005 / self.dt)
        self.inter_delay = int(0.005 / self.dt)
        self.delay_resp  = int(self.p["delay_resp"]  / self.dt)   
        self.resp_dur    = self.inter_delay 

        self.n_steps = 7
        self.n_inp, self.n_out = 2, 1
        self.step_block = self.cue_dur + self.inter_delay
        self.min_len    = self.n_steps*self.cue_dur + (self.n_steps-1)*self.inter_delay
        self.len        = dataset_len

        self.trial_len = self.min_len + self.delay_resp + self.resp_dur

    def __len__(self): return self.len

    def __getitem__(self, idx):
        return self._create_trial()

    # ────────────────────────────────────────────────────────────── #
    def _create_trial(self):
        c1_cnt = torch.randint(1, self.n_steps, ()).item()
        c2_cnt = self.n_steps - c1_cnt
        cue_labels = [0]*c1_cnt + [1]*c2_cnt
        random.shuffle(cue_labels)

        inp    = torch.zeros((self.trial_len, self.n_inp), dtype=torch.float32)
        target = torch.zeros((self.trial_len, self.n_out), dtype=torch.float32)

        t = 0
        for step, ch in enumerate(cue_labels):
            inp[t : t+self.cue_dur, ch] = 1.0
            t += self.cue_dur
            if step < self.n_steps-1:
                t += self.inter_delay

        # response window (same for every trial in this dataset)
        resp_start = t + self.delay_resp
        resp_end   = resp_start + self.resp_dur
        ans        = 1.0 if c1_cnt > c2_cnt else -1.0
        target[resp_start:resp_end] = ans
        mask   = torch.zeros_like(target) 
        mask  [resp_start:resp_end] = 1.0

        return inp, target, mask

class CosineContextDataset(Dataset):

    def __init__(
        self,
        task_params,
        dataset_len: int = 512,
        T: int = 500,                                   
        scale: float = 90.0,
        dtype=torch.float32,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.len = dataset_len
        self.T = T
        self.periods = tuple(
                p + task_params['delay_resp'] for p in (90, 105, 120, 135, 150, 165, 180)
            )
        self.scale = scale
        self.dtype = dtype

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.t = torch.arange(T, dtype=self.dtype)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        return self._create_trial()

    def _create_trial(self):
        period = random.choice(self.periods)
        amp = period / self.scale

        inp = torch.full((self.T, 1), amp, dtype=self.dtype)

        base_phase = 2 * math.pi * self.t / period
        target = torch.stack(
            (
                torch.cos(base_phase),                      # channel 0
                torch.cos(base_phase + math.pi / 4),        # channel 1 (+45°)
            ),
            dim=-1,                                         # (T, 2)
        )

        mask = torch.ones(self.T, 2, dtype=self.dtype)

        return inp, target, mask