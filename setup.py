from setuptools import setup

setup(
    name="neel",
    version="0.1.0",
    packages=["neel"],
    license="LICENSE",
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
        "accelerate",
    ],
)
