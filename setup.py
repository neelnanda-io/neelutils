from setuptools import setup

setup(
    packages=["neel"],
    description="Neel's personal utils - you're welcome to use, but this is very badly maintained and commented!.",
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "plotly",
        "rich",
        "matplotlib",
        "transformer_lens",
    ],
    dependency_links=[
        'git+https://github.com/neelnanda-io/neel-plotly.',
    ]
)
