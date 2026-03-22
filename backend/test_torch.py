import torch

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())