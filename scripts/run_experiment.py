#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json
import pandas as pd

from gsnoop.util import diff_transform, xor_transform
from gsnoop.causal import find_hitting_set, find_greedy_hitting_set
from gsnoop.screening import group_screening, lasso_screening

np.random.seed(1)

# retrieve task id from args
index = 0

# retrieve metadata

records = []
for repetition in range(repetitions):
    # generate x and y
    

    # compute x_diff and x_xor for screening
    x_diff, y_diff = diff_transform(x, y)
    x_xor, y_xor = xor_transform(x, y)
    
    # compute t=1 lasso baseline
    # compute lasso with most-stable L1 regularization
    # compute stepwise screening with most-stable L1 regularization
    # compute MHS with Hochbaum approximation
    
    # compute precision, recall, f1 score for all repetitions

# store results to f"./results/{index}.json"
with open(f'./results/{index}.json', 'w+') as f:
    pass