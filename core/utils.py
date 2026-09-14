"""Small helpers shared by training and evaluation."""

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def load_pickle(file_path: str | Path) -> Any:
    """Load a pickled object.

    Args:
        file_path (str | Path): File to read.

    Returns:
        Any: The unpickled object.
    """
    with open(file_path, "rb") as fh:
        return pickle.load(fh)
