#!/usr/bin/env python3

from ray_tune_cli import ray_tune_cli
from mnist_lightning_cli import lightning_cli


if __name__ == "__main__":
    ray_tune_cli(lightning_cli)
