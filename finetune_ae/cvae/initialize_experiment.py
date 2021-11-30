import numpy as np
import torch
import random


def initialize_random_seed(seed=3):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_for_early_break_condition(val_losses, required_window=5):
    if len(val_losses) < required_window:
        return False
    # return True if val_loss has strictly increased within the required window
    windowOI = val_losses[-required_window:]
    return min(windowOI) == windowOI[0]  # has been increasing
