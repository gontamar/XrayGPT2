"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import xraygpt.tasks as tasks
from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank, init_distributed_mode
from xraygpt.common.logger import setup_logger
from xraygpt.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from xraygpt.common.registry import registry
from xraygpt.common.utils import now

# imports modules for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank() # get_rank from xraygpt/common/dist_utils.py

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # Set seeds in stdlib, numpy, pytorch

    cudnn.benchmark = False 
    cudnn.deterministic = True # PyTorch backend config

#Uses the global registry (xraygpt/common/registry.py) that was populated by imports from runners, models, etc.
def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.

    job_id = now() # From xraygpt/common/utils.py

    cfg = Config(parse_args()) # Loads config and applies CLI overrides (xraygpt/common/config.py)

    init_distributed_mode(cfg.run_cfg) # xraygpt/common/dist_utils.py

    setup_seeds(cfg) # Calls function above (seeds randomness)

    # set after init_distributed_mode() to only log on master.

    setup_logger()    # xraygpt/common/logger.py

    cfg.pretty_print() # Pretty print config (xraygpt/common/config.py)

    task = tasks.setup_task(cfg) # xraygpt/tasks/__init__.py, will instantiate a task class
    datasets = task.build_datasets(cfg)  # Calls the method on task object, likely uses xraygpt/datasets/builders/
    model = task.build_model(cfg)  # Calls the method on task, likely uses xraygpt/models/

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()  # Calls train() method on the runner (from xraygpt/runners/)


if __name__ == "__main__":
    main()
