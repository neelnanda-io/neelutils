
import torch
import numpy as np
import einops
from transformer_lens.utils import to_numpy
from IPython.display import display
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

process_tokens = lambda l: [process_token(s) for s in l]
process_tokens_index = lambda l: [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

def cos(x, y):
    return (einops.einsum(x, y, "... a, ... a -> ...")) / x.norm(dim=-1) / y.norm(dim=-1)

def show_df(df):
    display(df.style.background_gradient("coolwarm"))