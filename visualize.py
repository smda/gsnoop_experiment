import pandas as pd
import os
import json

import matplotlib.pyplot as plt

files = [f"results/{p}" for p in os.listdir("results")]
print(len(files))

for fp in files:
    with open(fp, "r") as f:
        df = json.load(f)
        for record in df["metrics"]:
            print(record)
        break
