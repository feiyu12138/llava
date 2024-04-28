# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .coarser import Coarser
from .finer import Finer
from .formatter import Formatter
from .selector import Selector
from ....args import import_config


if __name__ == "__main__":
    config_file = 'cfgs/roberta/base-512/postnorm-16n.py'
    config = import_config(config_file)
    finer = Finer(config)
    coarser = Coarser(config)
    selector = Selector(config)
    formatter = Formatter(config)
    
