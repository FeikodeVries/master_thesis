import torch

x = torch.rand((2, 12288))

for i, j in enumerate(x):
    print(j.shape)
