import argparse
import numpy as np
import pandas as pd
import torch
import math
import random
import os
import sys
import datetime
import copy
import time


def set_random_seed(args):
    seed = args.seed
    if getattr(args, "deterministic", True):
        # must be set before any CUDA context is created (e.g. before torch.cuda.manual_seed_all)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if getattr(args, "deterministic", True):
        # covers torch-native ops (scatter_add_/index_add_/etc.) used e.g. in DRV feature maps
        # and timing slack aggregation, which are not affected by the custom CUDA kernels' own
        # --deterministic switch. warn_only=True so an op without a deterministic kernel just
        # prints a warning instead of aborting the run.
        torch.use_deterministic_algorithms(True, warn_only=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
