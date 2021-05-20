import sys
import os
import csv
import numpy as np



def decode_rpin_log(log_file):
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

    return header, data


def decode_logs(experiment_path):
    log_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(experiment_path) \
            for f in filenames if os.path.splitext(f)[1] == '.txt']

    logs = {}
    for log_file in log_files:
        try:
            logs[log_file] = decode_rpin_log(log_file)
        except:
            continue
    return logs


if __name__ == '__main__':
    #experiment_path = '/mnt/fs1/mrowca/dummy1/RPIN/'
    experiment_path = sys.argv[1]
    logs = decode_logs(experiment_path)
    print(logs)
