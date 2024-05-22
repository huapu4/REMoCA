# -*- coding: utf-8 -*-
"""
Provide p3d(63 layers, 131 layers)
@Time    : 2021/10/28
@Author  : Lin Zhenzhe, Zhang Shuyi
"""
from __future__ import print_function

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['P3D', 'P3D63', 'P3D131']

