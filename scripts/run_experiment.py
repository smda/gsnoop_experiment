#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import itertools
import grouplib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Pool, cpu_count
import collections

from sklearn.metrics import (
    mean_absolute_percentage_error,
    explained_variance_score,
)
from sklearn.linear_model import Lasso
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import csv
import os

# execfile("synthetic_functions.py")
exec(open("synthetic_functions.py").read())

with open("synthetic_metadata.json", "r") as f:
    metadata = json.load(f)


def compute_experiment(params):
    # parse metadata
    features = params["system"]["features"]
    repetition = params["repetition"]
    abs_sample_size = max(1, int(params["rel_sample_size"] * features))
    index = params["system"]["index"]
    relevant_terms_level = params["system"]["relevant_terms_level"]
    interaction_p = params["system"]["interaction_p"]

    relevant_options = []
    for term in params["system"]["terms"]:
        relevant_options += term["options"]
    func = systems[index]

    # prepare data
    X = np.unique(np.random.randint(2, size=(2 * abs_sample_size, features)), axis=1)
    Y = StandardScaler().fit_transform(np.array(list(map(func, X))).reshape(-1, 1))

    x_train, y_train = X[:abs_sample_size, :], Y[:abs_sample_size]
    x_test, y_test = X[abs_sample_size:, :], Y[abs_sample_size:]

    # pre-compute group sampling (takes long..)
    xg_train, yg_train = grouplib.diff_transform(x_train, y_train)

    # perform feature selection
    lasso_options = grouplib.lasso_screening(x_train, y_train)
    group_options = grouplib.group_screening(xg_train, yg_train)

    # compute precision, recall, f1 score and Jaccard similarity
    metrix = {
        "lasso_precision": grouplib.precision(relevant_options, lasso_options),
        "lasso_recall": grouplib.recall(relevant_options, lasso_options),
        "lasso_f1": grouplib.f1(relevant_options, lasso_options),
        "group_precision": grouplib.f1(relevant_options, group_options),
        "group_recall": grouplib.f1(relevant_options, group_options),
        "group_f1": grouplib.f1(relevant_options, group_options),
        "jaccard": grouplib.jaccard(group_options, lasso_options),
    }
    metrix.update(
        {
            "repetition": repetition,
            "features": features,
            "rel_sample_size": params["rel_sample_size"],
            "abs_sample_size": abs_sample_size,
            "relevant_terms_level": relevant_terms_level,
            "system": index,
            "relevant_options": len(relevant_options),
            "interaction_p": interaction_p,
        }
    )

    # Perform experiment for one data point
    max_t = 3
    lasso_models = [
        grouplib.twise_optimize(
            HalvingGridSearchCV, Lasso, x_train[:, lasso_options], y_train, t=t
        )
        for t in range(1, max_t + 1)
    ]
    group_models = [
        grouplib.twise_optimize(
            HalvingGridSearchCV, Lasso, x_train[:, group_options], y_train, t=t
        )
        for t in range(1, max_t + 1)
    ]

    # baseline lasso model
    baseline = grouplib.twise_optimize(
        HalvingGridSearchCV, Lasso, x_train, y_train, t=1
    )
    baseline_opts = np.array(
        list(sorted(np.where(baseline.best_estimator_.coef_ != 0)[0]))
    )
    metrix["baseline_precision"] = grouplib.precision(relevant_options, baseline_opts)
    metrix["baseline_recall"] = grouplib.recall(relevant_options, baseline_opts)
    metrix["baseline_f1"] = grouplib.f1(relevant_options, baseline_opts)

    metrix.update(
        {
            "baseline_mape": mean_absolute_percentage_error(
                y_test, baseline.predict(x_test)
            ),
            "baseline_expvar": explained_variance_score(
                y_test, baseline.predict(x_test)
            ),
        }
    )

    # evaluate models with regard to mape and explained variance
    for i, t in enumerate(range(1, max_t + 1)):
        if t > 1:
            xt_lasso = PolynomialFeatures(
                degree=t, interaction_only=True
            ).fit_transform(x_test[:, lasso_options])
            xt_group = PolynomialFeatures(
                degree=t, interaction_only=True
            ).fit_transform(x_test[:, group_options])
        else:
            xt_lasso = x_test[:, lasso_options]
            xt_group = x_test[:, group_options]

        metrix[f"lasso_mape-{t}"] = mean_absolute_percentage_error(
            y_test, lasso_models[i].predict(xt_lasso)
        )
        metrix[f"group_mape-{t}"] = mean_absolute_percentage_error(
            y_test, group_models[i].predict(xt_group)
        )

        metrix[f"lasso_expvar-{t}"] = explained_variance_score(
            y_test, lasso_models[i].predict(xt_lasso)
        )
        metrix[f"group_expvar-{t}"] = explained_variance_score(
            y_test, group_models[i].predict(xt_group)
        )

    return metrix


if __name__ == "__main__":
    pgrid = {
        "repetition": np.arange(30).tolist(),
        "system": metadata[:],
        "rel_sample_size": np.linspace(0.33, 2.33, 7).tolist(),
    }

    commands = []
    for i, p in enumerate(ParameterGrid(pgrid)):
        command = f"python3 run_experiment.py {i}\n"
        commands.append(command)

        with open(f"./screening_params/{i}.json", "w+") as f:
            json.dump(p, f)

    os.system("rm -rf ./screening_jobs.txt")
    with open("screening_jobs.txt", "w") as f:
        f.writelines(commands)
