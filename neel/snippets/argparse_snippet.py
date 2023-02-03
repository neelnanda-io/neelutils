import argparse
import json

DEFAULT_CFG = {
    "alpha": 123,
    "beta": True,
    "charlie": False,
    "delta": "143",
}


def arg_parse_update_cfg(default_cfg):
    """
    Helper function to
    """
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in DEFAULT_CFG.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    return cfg


cfg = arg_parse_update_cfg(DEFAULT_CFG, False)
print(json.dumps(cfg, indent=2))
print(json.dumps(DEFAULT_CFG, indent=2))
