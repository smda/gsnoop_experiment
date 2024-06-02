#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import json

'''
recision": {
        "baseline-normal": 0.3,
        "baseline-group": 1.0,
        "sizefit-normal": 1.0,
        "sizefit-group": 1.0,
        "causal-group": 0.0,
        "stepsize-normal-0.05": 1.0,
        "stepsize-group-0.05": 1.0,
        "stepsize-normal-0.025": 1.0,
        "stepsize-group-0.025": 0.75,
        "stepsize-normal-0.01": 0.42857142857142855,
        "stepsize-group-0.01": 0.3333333333333333,
        "stepsize-normal-0.005": 0.21428571428571427,
        "stepsize-group-0.005": 0.25
'''

def merge_data():
    files = ["results/" + d for d in os.listdir("results/")]

    df = []
    for file in files:
        with open(file, "r") as f:
            row = json.load(f)

            # aggregate
            metrics = {}
            for metric in ['precision', 'recall', 'f1']:
                for method in ['baseline', 'sizefit', 'causal', 'stepsize-0.05', 'stepsize-0.025', 'stepsize-0.01', 'stepsize-0.005']:
                    for typ in ['normal', 'group']:
                        if method == "causal" and typ == "normal": continue
                        colname = metric  + '_' + method + "_" + typ
                        metrics[colname] = []
                            

            for record in row["metrics"]:
                # precision
                metrics["precision_baseline_normal"].append(record['precision']['baseline-normal'])
                metrics["precision_baseline_group"].append(record['precision']['baseline-group'])
                metrics["precision_sizefit_normal"].append(record['precision']['sizefit-normal'])
                metrics["precision_sizefit_group"].append(record['precision']['sizefit-group'])
                metrics["precision_causal_group"].append(record['precision']['causal-group'])
                metrics["precision_stepsize-0.05_normal"].append(record['precision']['stepsize-normal-0.05'])
                metrics["precision_stepsize-0.05_group"].append(record['precision']['stepsize-group-0.05'])
                metrics["precision_stepsize-0.025_normal"].append(record['precision']['stepsize-normal-0.025'])
                metrics["precision_stepsize-0.025_group"].append(record['precision']['stepsize-group-0.025'])
                metrics["precision_stepsize-0.01_normal"].append(record['precision']['stepsize-normal-0.01'])
                metrics["precision_stepsize-0.01_group"].append(record['precision']['stepsize-group-0.01'])
                metrics["precision_stepsize-0.005_normal"].append(record['precision']['stepsize-normal-0.005'])
                metrics["precision_stepsize-0.005_group"].append(record['precision']['stepsize-group-0.005'])
                
                # recall
                metrics["recall_baseline_normal"].append(record['recall']['baseline-normal'])
                metrics["recall_baseline_group"].append(record['recall']['baseline-group'])
                metrics["recall_sizefit_normal"].append(record['recall']['sizefit-normal'])
                metrics["recall_sizefit_group"].append(record['recall']['sizefit-group'])
                metrics["recall_causal_group"].append(record['recall']['causal-group'])
                metrics["recall_stepsize-0.05_normal"].append(record['recall']['stepsize-normal-0.05'])
                metrics["recall_stepsize-0.05_group"].append(record['recall']['stepsize-group-0.05'])
                metrics["recall_stepsize-0.025_normal"].append(record['recall']['stepsize-normal-0.025'])
                metrics["recall_stepsize-0.025_group"].append(record['recall']['stepsize-group-0.025'])
                metrics["recall_stepsize-0.01_normal"].append(record['recall']['stepsize-normal-0.01'])
                metrics["recall_stepsize-0.01_group"].append(record['recall']['stepsize-group-0.01'])
                metrics["recall_stepsize-0.005_normal"].append(record['recall']['stepsize-normal-0.005'])
                metrics["recall_stepsize-0.005_group"].append(record['recall']['stepsize-group-0.005'])
                
                # f1
                metrics["f1_baseline_normal"].append(record['f1_score']['baseline-normal'])
                metrics["f1_baseline_group"].append(record['f1_score']['baseline-group'])
                metrics["f1_sizefit_normal"].append(record['f1_score']['sizefit-normal'])
                metrics["f1_sizefit_group"].append(record['f1_score']['sizefit-group'])
                metrics["f1_causal_group"].append(record['f1_score']['causal-group'])
                metrics["f1_stepsize-0.05_normal"].append(record['f1_score']['stepsize-normal-0.05'])
                metrics["f1_stepsize-0.05_group"].append(record['f1_score']['stepsize-group-0.05'])
                metrics["f1_stepsize-0.025_normal"].append(record['f1_score']['stepsize-normal-0.025'])
                metrics["f1_stepsize-0.025_group"].append(record['f1_score']['stepsize-group-0.025'])
                metrics["f1_stepsize-0.01_normal"].append(record['f1_score']['stepsize-normal-0.01'])
                metrics["f1_stepsize-0.01_group"].append(record['f1_score']['stepsize-group-0.01'])
                metrics["f1_stepsize-0.005_normal"].append(record['f1_score']['stepsize-normal-0.005'])
                metrics["f1_stepsize-0.005_group"].append(record['f1_score']['stepsize-group-0.005'])
                
                

            metrics = {k: np.mean(v) for k, v in metrics.items()}
            del row["metrics"]
            del row["relevant_options"]
            metrics.update(row)

            df.append(metrics)

    df = pd.DataFrame(df)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    merge_data()
