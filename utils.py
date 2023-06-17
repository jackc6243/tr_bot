import torch

def convert_to_tensor(x):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return [torch.tensor(i).to(device).type(torch.cuda.FloatTensor) for i in x]

