
import torch
import numpy as np
import einops
from transformer_lens.utils import to_numpy
from IPython.display import display, HTML
import pandas as pd
from neel_plotly import *
SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s, model=None):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

process_tokens = lambda l, model: [process_token(s, model) for s in l]
process_tokens_index = lambda l, model: [f"{process_token(s, model)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None, model=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)), model)
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

def cos(x, y):
    return (einops.einsum(x, y, "... a, ... a -> ...")) / x.norm(dim=-1) / y.norm(dim=-1)

def show_df(df):
    display(df.style.background_gradient("coolwarm"))


from html import escape
import colorsys

from IPython.display import display


def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()
    
    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))


# s = create_html(["a", "b\nd", "c        d"], [1, -2, -3])
# s = create_html(["a", "b\nd", "c        d", "e"], [1, -2, -3], allow_different_length=True)

def add_to_df(df, name, tensor):
    df[name] = to_numpy(tensor.flatten())
    return df

def show_df(df):
    display(df.style.background_gradient("coolwarm"))

def get_induction_scores(model, make_plot=False, batch_size=4, ind_seq_len = 200):

    rand_tokens_vocab = torch.tensor([i for i in range(1000, 10000) if "  " not in model.to_string(i)]).cuda()

    random_tokens = rand_tokens_vocab[torch.randint(0, len(rand_tokens_vocab), (batch_size, ind_seq_len))]
    bos_tokens = torch.full(
        (batch_size, 1), model.tokenizer.bos_token_id, dtype=torch.long
    ).cuda()
    ind_tokens = torch.cat([bos_tokens, random_tokens, random_tokens], dim=1)
    print("ind_tokens.shape", ind_tokens.shape)
    _, ind_cache = model.run_with_cache(ind_tokens)

    ind_head_scores = einops.reduce(
        ind_cache.stack_activation("pattern").diagonal(ind_seq_len-1, -1, -2),
        "layer batch head diag_pos -> layer head", "mean")
    if make_plot: imshow(ind_head_scores, xaxis="Head", yaxis="Layer", title="Induction Head Scores")
    return ind_head_scores

def make_neuron_df(n_layers, d_mlp):
    neuron_df = pd.DataFrame({
        "L": [l for l in range(n_layers) for n in range(d_mlp)],
        "N": [n for l in range(n_layers) for n in range(d_mlp)],
        "label": [f"L{l}N{n}" for l in range(n_layers) for n in range(d_mlp)],
    })
    return neuron_df