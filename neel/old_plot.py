# %%
# # Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops

import logging

import tqdm.notebook as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import partial
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
import sys
from pprint import pprint
import math

# import pysvelte
# from easy_transformer import EasyTransformer, HookedRootModule, HookPoint
# from rich import print
# %%
def to_numpy(tensor, flat=False):
    if type(tensor) != torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def imshow(
    tensor,
    xaxis=None,
    yaxis=None,
    animation_name="Snapshot",
    return_fig=False,
    **kwargs,
):
    # tensor = torch.squeeze(tensor)
    if "x" in kwargs:
        if len(kwargs["x"]) != len(set(kwargs["x"])):
            logging.warning(
                f"X axis labels are not unique, this will mess up imshow: {kwargs['x']}"
            )
    if "y" in kwargs:
        if len(kwargs["y"]) != len(set(kwargs["y"])):
            logging.warning(
                f"Y axis labels are not unique, this will mess up imshow: {kwargs['y']}"
            )
    fig = px.imshow(
        to_numpy(tensor, flat=False),
        labels={"x": xaxis, "y": yaxis, "animation_name": animation_name},
        **kwargs,
    )
    if return_fig:
        return fig
    else:
        fig.show()


# Set default colour scheme
imshow_pos = partial(imshow, color_continuous_scale="Blues")
# Creates good defaults for showing divergent colour scales (ie with both
# positive and negative values, where 0 is white)
imshow = partial(imshow, color_continuous_scale="RdBu", color_continuous_midpoint=0.0)


def line(y, x=None, hover=None, xaxis="", yaxis="", return_fig=False, **kwargs):
    if type(y) == torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x) == torch.Tensor:
        x = to_numpy(x, flat=True)
    fig = px.line(x=x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if return_fig:
        return fig
    else:
        fig.show()


def scatter(x, y, hover=None, return_fig=False, yaxis="", xaxis="", **kwargs):
    fig = px.scatter(
        x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), hover_name=hover, **kwargs
    )
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if return_fig:
        return fig
    else:
        fig.show()


def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    hover=None,
    return_fig=False,
    **kwargs,
):
    # Helper function to plot multiple lines
    if type(lines_list) == torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line) == torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(
            go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs)
        )
    if log_y:
        fig.update_layout(yaxis_type="log")
    if return_fig:
        return fig
    else:
        fig.show()


def line_marker(x, **kwargs):
    lines([x], mode="lines+markers", **kwargs)


def animate_lines(
    lines_list,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    xaxis="x",
    yaxis="y",
    return_fig=False,
    **kwargs,
):
    if type(lines_list) == list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    print(df)
    fig = px.line(
        df,
        x=xaxis,
        y=yaxis,
        animation_frame=snapshot,
        range_y=[lines_list.min(), lines_list.max()],
        hover_name=hover,
        **kwargs,
    )
    if return_fig:
        return fig
    else:
        fig.show()


def animate_multi_lines(
    lines_list,
    y_index=None,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    swap_y_animate=False,
    return_fig=False,
    **kwargs,
):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list) == list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j]) + [snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index + [snapshot, "x"])
    fig = px.line(
        df,
        x="x",
        y=y_index,
        animation_frame=snapshot,
        range_y=[lines_list.min(), lines_list.max()],
        hover_name=hover,
        **kwargs,
    )
    if return_fig:
        return fig
    else:
        fig.show()


def animate_scatter(
    lines_list,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    yaxis="y",
    xaxis="x",
    color=None,
    color_name="color",
    return_fig=False,
    **kwargs,
):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list) == list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color) == torch.Tensor:
        color = to_numpy(color)
    if len(color.shape) == 1:
        color = einops.repeat(color, "x -> snapshot x", snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(
                [
                    lines_list[i, 0, j].item(),
                    lines_list[i, 1, j].item(),
                    snapshot_index[i],
                    color[i, j],
                ]
            )
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    fig = px.scatter(
        df,
        x=xaxis,
        y=yaxis,
        animation_frame=snapshot,
        range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()],
        range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()],
        hover_name=hover,
        color=color_name,
        **kwargs,
    )
    if return_fig:
        return fig
    else:
        fig.show()


# %%
def loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def per_token_loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs


# %%
def add_button(fig, button, auto_pos=True):
    # Technically acts on update menus, not buttons
    if auto_pos:
        num_prev_buttons = len(fig.layout.updatemenus)
        button["y"] = 1 - num_prev_buttons * 0.15
    if "x" not in button:
        button["x"] = -0.1
    fig.update_layout(updatemenus=fig.layout.updatemenus + (button,))
    return fig


def add_axis_toggle(fig, axis, auto_pos=True):
    assert axis in "xy", f"Invalid axis: {axis}"
    is_already_log = fig.layout[f"{axis}axis"].type == "log"
    toggle_axis = dict(
        type="buttons",
        active=0 if is_already_log else -1,
        buttons=[
            dict(
                label=f"Log {axis}-axis",
                method="relayout",
                args=[{f"{axis}axis.type": "log"}],
                args2=[{f"{axis}axis.type": "linear"}],
            )
        ],
    )
    return add_button(fig, toggle_axis, auto_pos=auto_pos)


def axis_log(fig, axis):
    fig.layout[f"{axis}axis"].type = "log"
    return fig
