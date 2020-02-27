"""
Initialize stuff
"""

import pdb
from pathlib import Path
from typing import List, Dict, Any, Union
from yacs.config import CfgNode as CN
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import yaml
import re


Fpath = Union[Path, str]
Cft = CN
Arr = Union[np.array, List, torch.tensor]
# Ds = Dataset

# This is required to read 5e-4 as a float rather than string
# at all places yaml should be imported from here
yaml.SafeLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

# _SCRIPTPATH_ =
sys.path.append('./code/')
sys.path.append('./utils')


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
