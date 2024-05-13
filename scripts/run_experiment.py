#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json
import pandas as pd

from gsnoop.util import diff_transform, xor_transform, precision, recall, f1
from gsnoop.causal import find_hitting_set, find_greedy_hitting_set
from gsnoop.screening import group_screening, lasso_screening

REPETITIONS = 50

# no random here
np.random.seed(1)


def main(index):
    with open("./build/oracles.json", "r") as f:
        metadata = json.load(f)

    with open(f"./build/params/{index}.json", "r") as f:
        params = json.load(f)

    index = params["system"]
    features = metadata[index]["features"]
    system = params["system"]

    # experiment_repetition_id, not the repetition for each datapoint
    repetition = params["repetition"]
    abs_sample_size = max(1, int(params["rel_sample_size"] * features))

    relevant_terms_level = metadata[index]["relevant_terms_level"]
    interaction_p = metadata[index]["p_interaction_degree"]

    relevant_options = []
    for term in metadata[index]["terms"]:
        relevant_options += term["options"]

    # load oracles
    exec(open("./build/oracles.py").read())  # systems = ...

    records = []
    feature_selections = []
    for repetition in range(REPETITIONS):
        x = np.random.randint(2, size=(abs_sample_size, features))
        y = np.array(list(map(systems[index], x)))

        # compute x_diff and x_xor for screening
        x_diff, y_diff = diff_transform(x, y)
        x_xor, y_xor = xor_transform(x, y)

        # compute and store feature selections
        feature_selection = {
            "lasso_screen": lasso_screening(x_diff, y_diff),
            "group_screen": group_screening(x_diff, y_diff),
            "causal_screen": find_greedy_hitting_set(x_xor, y_xor, threshold=0.1),
        }
        feature_selections.append(feature_selection)

        # compute precision, recall, f1 score for all repetitions
        metrics = {
            "precision": {
                a: precision(relevant_options, feature_selection[a])
                for a in feature_selection.keys()
            },
            "recall": {
                a: recall(relevant_options, feature_selection[a])
                for a in feature_selection.keys()
            },
            "f1_score": {
                a: f1(relevant_options, feature_selection[a])
                for a in feature_selection.keys()
            },
        }

        # store options and classification metrics
        records.append(metrics)

    # merge everythin into a single dict
    results = {}
    results.update(
        {
            "system": index,
            "repetition": repetition,
            "features": features,
            "rel_sample_size": params["rel_sample_size"],
            "abs_sample_size": abs_sample_size,
            "relevant_terms_level": relevant_terms_level,
            "interaction_p": interaction_p,
            "relevant_options": relevant_options,
            "feature_selection": feature_selections,
            "metrics": records,
        }
    )
    return results


if __name__ == "__main__":
    import sys

    index = sys.argv[1]

    result = main(index)

    # store results
    with open(f"./results/{index}.json", "w+") as f:
        f.write(json.dumps(result, indent=2))
