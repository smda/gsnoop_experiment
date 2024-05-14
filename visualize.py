import pandas as pd
import os
import json

files = [f"results/{p}" for p in os.listdir("results")]
print(len(files))

for fp in files:
	with open(fp, "r") as f:
		df = json.load(f)
		print(df)
