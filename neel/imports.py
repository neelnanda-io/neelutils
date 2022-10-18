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
import tqdm.auto as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

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
import accelerate

from accelerate import Accelerator
from accelerate.utils import set_seed, write_basic_config
from accelerate import notebook_launcher
import os

from pprint import pprint

from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
import easy_transformer
# import pysvelte
from easy_transformer import EasyTransformerConfig
from easy_transformer import EasyTransformer
from easy_transformer.utils import lm_cross_entropy_loss, tokenize_and_concatenate, to_numpy, get_corner
import neel
from rich import print as rprint
from collections import OrderedDict
import logging
import re

from neel.plot import *
import neel.utils as utils
import neel.fourier as fourier

from collections import defaultdict

# %%
from IPython import get_ipython
try:
    ipython = get_ipython()
    IN_IPYTHON = True
    # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    print("In IPython")
    print("Set autoreload")
except:
    IN_IPYTHON = False
    print("Not in IPython")