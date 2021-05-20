import csv
import numpy as np

log_file = '/mnt/fs4/mrowca/hyperexamples/11000/RPIN/dominoes/0/model/log-050714-1799694.txt'

header = np.array([['steps', 'mean', 'p_1', 'p_2', 'o_1', 'o_2']])

with open(log_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    csv_data = [[r.strip() for r in row] for row in reader]

# Remove invalid rows
data = []
for row in csv_data:
    try:
        float(row[1])
        data.append([float(row[0][-5:-1])] + [float(r) for r in row[1:]])
    except:
        continue

data = np.array(data)
header_data = np.concatenate([header, data], axis=0)
