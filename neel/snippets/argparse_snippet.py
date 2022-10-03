import argparse

parser = argparse.ArgumentParser()

cfg = {
"d_model": 512,
"n_layers": 4,
"lr": 4e-4,
"batch_size": 32,
"batches_per_step": 3,
"seed": 98742,
    # 'checkpoint_every_tokens':5*10**7,
    "use_checkpoint_schedule": True,
    "debug": False,
    "debug_batch": False,
    "debug_overfit": False,
    "normalization": "LN",  # 'LN' 'RMS' or None
    # "max_tokens": 15 * 10 ** 9,
    "max_tokens": 22*10**9,
    "version": 32,
    "use_float16": False,
    "use_bfloat16": False,
    "save_checkpoints_to_bfloat16": True,
    "use_bfloat16_matmul": False,
    "right_multiply_matrices": True,
    # 'n_heads':8,
    "d_head": 128,
    "n_ctx": 1024,
    "d_vocab": 50278,
    # 'factor_size':256,
    "betas": (0.9, 0.99),
    "weight_decay": 0.01,
    "dataset_name": "the_pile",
    # "dataset_name": "wikipedia",
    # "dataset_subset_name": "20220301.en",
    "grad_norm_clip": 1.0,
    "use_attn_result": False,
    "n_devices": 8,
    "act_fn": "SoLU",
    "use_pos_resid": True,
    "attn_only": False,
    "ln_eps": 1e-5,
    "lr_schedule": "cosine_warmup",
    # "warmup_tokens": 25 * 10 ** 7,
    "warmup_tokens": 2*10**8,
    "factored_embed": False,
    "train_loss_ewma_beta": 0.99,
    "shuffled_data": True,
    "use_ET": False,
    # 'W_O_init_scale':True,
}
# accelerator.print('Old')
# accelerato(cfg)
# print()
cfg["n_heads"] = cfg["d_model"] // cfg["d_head"]
cfg["d_mlp"] = 4 * cfg["d_model"]
cfg["tokens_per_step"] = cfg["batch_size"] * \
    cfg["n_ctx"] * cfg["batches_per_step"]
cfg["max_steps"] = cfg["max_tokens"] // cfg["tokens_per_step"] * 8
cfg["warmup_steps"] = cfg["warmup_tokens"] // cfg["tokens_per_step"]
# cfg['checkpoint_every'] = cfg['checkpoint_every_tokens']//cfg['tokens_per_step']
if cfg["debug"] and not cfg["debug_overfit"]:
    # print("Old max steps:", cfg["max_steps"])
    cfg["max_steps"] = 25
cfg["n_params"] = 12 * cfg["n_layers"] * cfg["d_model"] ** 2
print()
print(cfg)

print()
for key in cfg:
    parser.add_argument("--" + key, type=type(cfg[key]), default=cfg[key])
args = parser.parse_args()
cfg.update(args.__dict__)
print(cfg)
