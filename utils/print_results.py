import sys
import pandas as pd

path = sys.argv[1]
data = pd.read_csv(path)
del data['Description']
del data['Seed']
del data['Readout Train Data']
del data['Rollout Length']
print(data)
