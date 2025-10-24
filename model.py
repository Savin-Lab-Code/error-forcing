import torch
import torch.nn as nn
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from RNN_cells import *
from utils import seed_everything

class RNN(nn.Module):
    def __init__(self, params):
        """
        Initializes a continuous time recurrent neural network model
        """

        super(RNN, self).__init__()
        self.params = params
        seed_everything(params["seed"])

        # standard full rank RNN
        if params['cell_type'] == 'RNN':
            self.cell = RNNCell(params)
        if params['cell_type'] == 'hRNN': #for RFLO
            self.cell = hRNNCell(params)

        self.x0 = torch.zeros(1, params["n_rec"], dtype=torch.float32)

    def forward(self, input, y, mask, pred=False, x0=None):
        """
        Do a forward pass through all time steps
        """
        batch_size = input.size(0)
        seq_len = input.size(1)

        # allocate tensors for hidden state and output
        outputs = torch.zeros(
            batch_size,
            seq_len,
            self.params["n_out"],
            device=self.cell.w_out.device,
            dtype=torch.float32,
        )

        hidden = torch.zeros(
            batch_size,
            seq_len + 1,
            self.params["n_rec"],
            device=self.cell.w_out.device,
            dtype=torch.float32,
        )

        hidden_act = torch.zeros(
            batch_size,
            seq_len + 1,
            self.params["n_rec"],
            device=self.cell.w_out.device,
            dtype=torch.float32,
        )

        h = 0*(torch.randn(batch_size, self.params["n_rec"], device=self.cell.w_out.device, dtype=torch.float32))
        hidden_act[:, 0] = h
        e_p = 0
        a_list = torch.zeros_like(hidden)
        # run through all timesteps
        for i, input_t in enumerate(input.split(1, dim=1)):
            h, output, e_p = self.cell(input_t.squeeze(dim=1), h, y, mask, i, e_p, pred)
            outputs[:, i] = output
            hidden_act[:, i + 1] = h
            a_list[:, i + 1] = e_p
        return hidden, hidden_act, outputs, a_list


def predict(
    rnn, input, target, mask, pred, loss_fn=None, _mask=None, x0=None, return_loss=False
):

    # disable gradients
    rnn.eval()
    
    device = rnn.cell.w_out.device
    input = input.to(device=device)

    with torch.no_grad():
        shp1 = input.shape[0]
        shp2 = input.shape[1]
        y = torch.zeros((shp1, shp2, rnn.cell.w_out.shape[1])) if pred else target
        rates, hidden_act, predict, a_list = rnn(input, y, mask, pred=pred, x0=x0)
        loss = torch.sum((predict - target)**2 * mask) / torch.sum(mask)
        print("test loss:", loss.item())
        print("==========================")
    return rates.cpu().detach().numpy(), hidden_act.cpu().detach().numpy(), predict.cpu().detach().numpy(), a_list.cpu().detach().numpy(), loss.item()