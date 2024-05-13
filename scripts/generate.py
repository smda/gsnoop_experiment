#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import itertools

from sklearn.model_selection import ParameterGrid

# determinisism ..
np.random.seed(0)

# load configs
from config import EXPERIMENT_CONFIG as conf
from config import SCREENING_CONFIG as sconf
from config import FEATURE_SIZE_CONFIG as fconf

parameters = ParameterGrid(conf)

strings = []
tdata = []
for i, c in enumerate(ParameterGrid(conf)):
    # generate lambda expressions
    features = c["features"]

    nterms = fconf["absolute_relevant_terms"][features][c["relevant_terms_level"]]

    interaction_degree = np.random.geometric(c["p_interaction_degree"], size=nterms)

    effects = np.random.laplace(0, c["effect_spread"], size=nterms)

    terms = [
        {
            "options": np.random.choice(features, size=interaction_degree[i]).tolist(),
            "effect": effects[i],
        }
        for i in range(nterms)
    ]

    s = (
        f"    {i}: lambda x: "
        + " + ".join(
            [
                str("*".join(["x[" + str(o) + "]" for o in t["options"]]))
                + "*"
                + str(t["effect"])
                for t in terms
            ]
        )
        + ",\n"
    )
    strings.append(s)
    tdata.append(
        {
            "index": i,
            "features": c["features"],
            "relevant_terms_level": c["relevant_terms_level"],
            "p_interaction_degree": c["p_interaction_degree"],
            "experiment_repetition_id": c["experiment_repetition_id"],
            "terms": terms,
        }
    )

print(" " * 11 + "-- writing oracles to file.")
with open("build/oracles.py", "w+") as f:
    f.writelines(["systems = {\n"] + strings + ["}\n"])

print(" " * 11 + "-- writing oracle metadata to file.")
with open("build/oracles.json", "w+") as f:
    f.write(json.dumps(tdata, indent=2))

print(" " * 11 + "-- generating parameterizations for experiment runs.")
parameters = {
    "repetition": list(range(len(conf['experiment_repetition_id'])),
    "system": list(range(len(tdata))),
    "rel_sample_size": sconf["relative_sample_size"],
}

commands = []
for i, p in enumerate(ParameterGrid(parameters)):
    command = f"python3 ./build/run_experiment.py {i}\n"
    commands.append(command)
    with open(f"./build/params/{i}.json", "w+") as f:
        json.dump(p, f)

print(" " * 11 + f"-- writing {len(commands)} commands to compute.")
with open("./build/jobs.txt", "w") as f:
    f.writelines(commands)
