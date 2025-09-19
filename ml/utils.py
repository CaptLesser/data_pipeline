from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int] = 42) -> int:
    s = int(seed if seed is not None else 42)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch  # type: ignore

        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(s)
    return s

