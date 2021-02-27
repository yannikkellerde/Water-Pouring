import pandas as pd
import numpy as np
from collections import defaultdict
df = pd.read_csv("volumes_processed.csv",index_col=0)
glasses = defaultdict(list)
for col in df.columns:
    glasses[col.split("_")[1]].append(col)

for key in glasses:
    max_list = np.array([max(df[col_name]) for col_name in glasses[key]])
    max_list = max_list[~np.isnan(max_list)]
    if len(max_list)==0:
        continue
    print(f"Glass {key}, max fill-level: {max(max_list)} ml, min fill-level: {min(max_list)}, avg fill-level: {sum(max_list)/len(max_list)}")
