import sys
import os
import csv
import pickle
import numpy as np


def get_readout_type(readout):
    if readout['model_fn'] == 'visual_scene_model_fn' and \
            readout['inp_time_steps'][0] != readout['val_time_steps'][0] and \
            readout['inp_time_steps'][1] == readout['val_time_steps'][1]:
        return 'A', 'entire sequence visually encoded, ' + \
                'label over entire sequence'
    elif readout['model_fn'] == 'rollout_scene_model_fn' and \
            readout['inp_time_steps'][0] != readout['val_time_steps'][0] and \
            readout['inp_time_steps'][1] == readout['val_time_steps'][1]:
        return 'B', 'seen frames visually encoded, unseen frames predicted, ' + \
                'label over entire sequence'
    elif readout['model_fn'] == 'visual_scene_model_fn' and \
            readout['inp_time_steps'][0] != readout['val_time_steps'][0] and \
            readout['inp_time_steps'][1] != readout['val_time_steps'][1]:
        return 'C', 'seen visual frames visually encoded, unseen frames ignored, ' + \
                'label over entire sequence'
    elif readout['model_fn'] == 'visual_scene_model_fn' and \
            readout['inp_time_steps'][0] == readout['val_time_steps'][0] and \
            readout['inp_time_steps'][1] == readout['val_time_steps'][1]:
        return 'D', 'seen frames visually encoded, unseen frames ignored, ' + \
                'label over seen frames only'
    else:
        raise ValueError("Unknown Readout Type! %s" % readout)


def parse_result(result, subsample_factor = 6):
    seed = result['seed']
    model = result['model_dir'].split('/')[-5]
    train = result['train_name']
    readout_train = result['train_feature_file'].split('/')[-2].replace('train_', '')
    readout_test = result['test_feature_file'].split('/')[-2].replace('test_', '')
    data = []
    for readout in result['results']:
        readout_type, description = get_readout_type(readout)
        data.append({
            'Seed': seed,
            'Model': model,
            'Model Train Data': train,
            'Readout Train Data': readout_train,
            'Readout Test Data': readout_test,
            'Train Accuracy': readout['result']['train_accuracy'],
            'Test Accuracy': readout['result']['test_accuracy'],
            'Readout Type': readout_type,
            'Sequence Length': (readout['val_time_steps'][1] + 1) * subsample_factor,
            'Readout Train Positive': readout['result']['num_train_pos'],
            'Readout Train Negative': readout['result']['num_train_neg'],
            'Readout Test Positive': readout['result']['num_test_pos'],
            'Readout Test Negative': readout['result']['num_test_neg'],
            'Description': description,
            })

    return data

def combine_results(experiment_path):
    result_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(experiment_path) \
            for f in filenames if f == 'metrics_results.pkl']

    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            result = parse_result(result)
            results.extend(result)
        except:
            print('Could not read file %s' % result_file)
            continue
    return results


def write_csv(results, path, file_name = 'results.csv'):
    if len(results) < 1:
        print('No results found: %s' % path)
        return

    file_path = os.path.join(path, file_name)
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print('%d results written to %s' % (len(results), file_path))

if __name__ == '__main__':
    #experiment_path = '/mnt/fs1/mrowca/test_all'
    experiment_path = sys.argv[1]
    results = combine_results(experiment_path)
    write_csv(results, experiment_path)
