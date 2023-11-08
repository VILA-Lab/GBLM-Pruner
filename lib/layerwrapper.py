import torch
import torch.nn as nn
import os

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # self.scaler_row_2 = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
        # self.activations = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        # self.scaler_row_2 *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        # self.scaler_row_2 += torch.sum(inp, dim=1) / self.nsamples
        # uid = f"{self.layer_id}_{self.layer_name}"
        # print(f"Done getting the activation for {uid} sample no {self.nsamples}")
        # if self.layer_id < 19:
        #     return
        # file_path = f"/l/users/rocktim.jyotidas/wanda/wanda/activations4/{uid}.pth"
        # if os.path.exists(file_path):
        #     print(f"The file {file_path} exists.")
        #     activation_list = torch.load(file_path, map_location=torch.device('cpu'))
        # else:
        #     print(f"The file {file_path} does not exist.")
        #     activation_list = []
        # cpu_copy = inp.to(torch.device('cpu'))
        # activation_list.append(cpu_copy)
        # torch.save(activation_list, file_path)
        # self.activations.append(cpu_copy)
        