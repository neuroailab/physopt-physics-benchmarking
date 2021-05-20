import csv
import numpy as np

log_file = '/mnt/fs1/mrowca/test11/SVG/cloth/0/model/logs/log.txt'

header = np.array([['epoch', 'mse loss', 'kld loss', 'steps']])

with open(log_file, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    csv_data = [[r.strip() for r in row] for row in reader]

data = []
for row in csv_data:
    data.append([
        float(row[0][1:-1]),
        float(row[3]),
        float(row[7]),
        float(row[8][1:-1]),
        ])

data = np.array(data)
header_data = np.concatenate([header, data], axis=0)
