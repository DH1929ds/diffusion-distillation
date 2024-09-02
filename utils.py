import random
import numpy as np
import torch

def set_seed(seed):
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    
    # For CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure that the seed makes everything deterministic (potentially at the cost of execution speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False