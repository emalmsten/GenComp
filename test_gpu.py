import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.current_device())  # Displays the current GPU device
print(torch.cuda.get_device_name(0))  # Gets the name of the GPU being used