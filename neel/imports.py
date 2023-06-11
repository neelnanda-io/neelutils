# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import einops

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook_connected"
# pio.renderers.default = "colab"s
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import transformers
from datasets import load_dataset
import json
from transformers import AutoTokenizer
import transformers
import datasets
import time
import wandb

# import accelerate

# from accelerate import Accelerator
# from accelerate.utils import set_seed, write_basic_config
# from accelerate import notebook_launcher
import os

from pprint import pprint

from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional, Callable
from typing_extensions import Literal
# import circuitsvis as cv
# import pysvelte

import transformer_lens
# import pysvelte
from transformer_lens import HookedTransformerConfig, HookedTransformer, FactoredMatrix, ActivationCache
from transformer_lens.utils import (
    lm_cross_entropy_loss,
    tokenize_and_concatenate,
    to_numpy,
    get_corner,
    get_act_name,
    remove_batch_dim,
)
import transformer_lens.evals as evals
from transformer_lens.hook_points import HookedRootModule, HookPoint
# from transformer_lens.torchtyping_helper import T

try:
    import transformer_lens.loading_from_pretrained as loading
except:
    print("Loading from pretrained not present")
    pass
import transformer_lens.utils as utils
# import neel
from rich import print as rprint
from collections import OrderedDict
import logging
import re

# from neel.plot import *
import neel.utils as nutils
# import neel.fourier as fourier

from collections import defaultdict

# import gradio as gr

from IPython.display import HTML

# from fancy_einsum import einsum
# from torchtyping import TensorType as TT
import pprint
# import solu.utils as sutils
# %%
from IPython import get_ipython


def activate_autoreload():
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    print("In IPython")
    print("Set autoreload")


ipython = get_ipython()
if ipython is not None:
    print("In IPython")
    IN_IPYTHON = True
    activate_autoreload()
    # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
    import tqdm.notebook as tqdm
else:
    print("Not in IPython")
    IN_IPYTHON = False
    import tqdm

print("Imported everything!")
