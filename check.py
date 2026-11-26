import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("Total GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
x = torch.randn(1000, 1000, device="cuda")
print((x @ x).sum().item())