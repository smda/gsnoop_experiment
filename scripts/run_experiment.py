#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import json

from multiprocessing import Pool

from gsnoop.util import diff_transform, xor_transform, precision, recall, f1
from gsnoop.screening import (
    baseline_screening,
    stable_screening,
    stepwise_screening,
    find_greedy_hitting_set,
)

exec(open("./build/oracles.py").read())

REPETITIONS = 30

# no random here
np.random.seed(1)

#import warnings
#warnings.filterwarnings("ignore")

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
    # feature_selections = []
    for repetition in range(REPETITIONS):
        x = np.random.randint(2, size=(abs_sample_size, features))
        y = np.array(list(map(systems[index], x)))

        # compute x_diff and x_xor for screening
        x_diff, y_diff = diff_transform(x, y)
        x_xor, y_xor = xor_transform(
            x, y
        )  # threshold is meaningless here, since we do not add any noise
        x_xor = np.vstack([x_xor[i, :] for i in range(x_xor.shape[0]) if y_xor[i] != 0])

        '''
        # compute and store feature selections
        feature_selection = {
            "baseline-normal": baseline_screening(x, y),
            "baseline-group": baseline_screening(x, y),
            "sizefit-normal": stable_screening(x, y_diff),
            "sizefit-group": stable_screening(x_diff, y_diff),
            "causal-group": find_greedy_hitting_set(x_xor),
        }
        '''
        tolerances = [0.05, 0.025, 0.01]
        stepwise_options_normal = stepwise_screening(x, y, tolerances)
        print(stepwise_options_normal)
        #stepwise_options_group = stepwise_screening(x_diff, y_diff, tolerances)

        for k, tolerance in enumerate(tolerances):
            feature_selection[f"stepsize-normal-{tolerance}"] = stepwise_options_normal[
                k
            ]
            #feature_selection[f"stepsize-group-{tolerance}"] = stepwise_options_group[
            #    k
            #]

        # feature_selections.append(feature_selection)

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
        #records.append(metrics)
        
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
            # "feature_selection": feature_selections,
            "metrics": records,
        }
    )
    return results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def process_index(index):
    print(index)
    result = main(index)
    with open(f"results/{index}.json", "w") as f:
        f.write(json.dumps(result, indent=2, cls=NpEncoder))


if __name__ == "__main__":
    indices = range(100)#range(len(os.listdir('./build/params/'))) 
    print(f'Starting experiment with {len(list(indices))} runs.')
    for index in indices:
        process_index(index)
    #with Pool(processes=4) as pool: 
    #    pool.map(process_index, indices)
