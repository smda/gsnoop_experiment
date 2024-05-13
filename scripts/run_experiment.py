#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json
import pandas as pd

from gsnoop.util import diff_transform, xor_transform, precision, recall, f1_score
from gsnoop.causal import find_hitting_set, find_greedy_hitting_set
from gsnoop.screening import group_screening, lasso_screening

from .scripts.config import REPETITIONS

# no random here
np.random.seed(1)

# retrieve task id from args
index = 0

# TODO retrieve
relevant_options = []

# TODO retrieve and compute metadata sucha s absolute feature size

# retrieve metadata
with open(f"./build/params/{index}.json", "r") as f:
    params = json.load(f)

# load oracles
exec(open("./build/oracles.py").read()) # systems = ...

records = []
feature_selections = []
for repetition in range(REPETITIONS):

    # TODO generate x and y

    # compute x_diff and x_xor for screening
    x_diff, y_diff = diff_transform(x, y)
    x_xor, y_xor = xor_transform(x, y)

	# compute and store feature selections
    feature_selection = {
		'lasso_screen': lasso_screening(x_diff, y_diff),
		'group_screen': group_screening(x_diff, y_diff),
		'causal_screen': find_greedy_hitting_set(x_xor, y_xor, threshold=0.1)
	}
	feature_selections.append(feature_selection)

    # compute precision, recall, f1 score for all repetitions
	metrics = {
		'precision': {
			a: precision(relevant_options, feature_selection[a]) for a in feature_selection.keys()
		},
		'recall': {
			a: recall(relevant_options, feature_selection[a]) for a in feature_selection.keys()
		},
		'f1_score': {
			a: f1(relevant_options, feature_selection[a]) for a in feature_selection.keys()
		},


	# store options and classification metrics
	records.append(metrics)

# store results to f"./results/{index}.json"
with open(f"./results/{index}.json", "w+") as f:
    pass
