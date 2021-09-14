import sys
import os
import csv
import pickle
import numpy as np
import traceback

def get_model_attributes(model, train, seed):
    if model == 'CSWM':
        return {
            'Encoder Type': 'CSWM encoder',
            'Dynamics Type': 'CSWM dynamics',
            'Encoder Pre-training Task': 'null', 
            'Encoder Pre-training Dataset': 'null', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'Contrastive',
            'Encoder Training Dataset': train, 
            'Encoder Training Seed': seed, 
            'Dynamics Training Task': 'Contrastive',
            'Dynamics Training Dataset': train, 
            'Dynamics Training Seed': seed, 
            }
    elif model[-10:] == 'FrozenLSTM':
        return {
            'Encoder Type': model[:-10],
            'Dynamics Type': 'LSTM',
            'Encoder Pre-training Task': 'ImageNet classification', 
            'Encoder Pre-training Dataset': 'ImageNet', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'null',
            'Encoder Training Dataset': 'null', 
            'Encoder Training Seed': 'null', 
            'Dynamics Training Task': 'L2 on latent',
            'Dynamics Training Dataset': train, 
            'Dynamics Training Seed': seed, 
            }
    elif model[-9:] == 'FrozenMLP':
        return {
            'Encoder Type': model[:-9],
            'Dynamics Type': 'MLP',
            'Encoder Pre-training Task': 'ImageNet classification', 
            'Encoder Pre-training Dataset': 'ImageNet', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'null',
            'Encoder Training Dataset': 'null', 
            'Encoder Training Seed': 'null', 
            'Dynamics Training Task': 'L2 on latent',
            'Dynamics Training Dataset': train, 
            'Dynamics Training Seed': seed, 
            }
    elif model == 'OP3':
        return {
            'Encoder Type': 'OP3 encoder',
            'Dynamics Type': 'OP3 dynamics',
            'Encoder Pre-training Task': 'null', 
            'Encoder Pre-training Dataset': 'null', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'Image Reconstruction',
            'Encoder Training Dataset': train, 
            'Encoder Training Seed': seed, 
            'Dynamics Training Task': 'Image Reconstruction',
            'Dynamics Training Dataset': train, 
            'Dynamics Training Seed': seed, 
            }
    else:
        raise NotImplementedError

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
    output = {}
    for readout in result['results']:
        readout_type, description = get_readout_type(readout)
        count = 0
        processed = set()
        for i in range(len(readout['result']['labels'])):
            if readout['result']['stimulus_name'][i] in processed:
                print('Duplicated item: {}'.format(readout['result']['stimulus_name'][i]))
            else:
                count += 1
                processed.add(readout['result']['stimulus_name'][i])
                data = {
                    'Model': model,
                    'Readout Train Data': readout_train,
                    'Readout Test Data': readout_test,
                    'Train Accuracy': readout['result']['train_accuracy'],
                    'Test Accuracy': readout['result']['test_accuracy'],
                    'Readout Type': readout_type,
                    'Predicted Prob_false': readout['result']['test_proba'][i][0],
                    'Predicted Prob_true': readout['result']['test_proba'][i][1],
                    'Predicted Outcome': np.argmax(readout['result']['test_proba'][i]),
                    'Actual Outcome': readout['result']['labels'][i],
                    'Stimulus Name': readout['result']['stimulus_name'][i],
                    # 'Sequence Length': readout['val_time_steps'][1] * subsample_factor,
                    }
                data.update(get_model_attributes(model, train, seed))
                if model in output:
                    output[model].append(data)
                else:
                    output[model] = [data]
        print('Model: {}, Train: {}, Test: {}, Type: {}, Len: {}'.format(model, train, readout_test, readout_type, count))
    return output

def combine_results(experiment_path):
    result_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(experiment_path) \
            for f in filenames if f == 'metrics_results.pkl']
    print('\n'.join(result_files))

    results = {}
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            result = parse_result(result)
            results = merge_results(results, result)
        except Exception:
            print('Could not read file %s' % result_file)
            traceback.print_exc()
            continue
    return results

def merge_results(old_results, new_results):
    for model, data in new_results.items():
        if model in old_results:
            old_results[model] += data
        else:
            old_results[model] = data
    return old_results

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
    experiment_path = sys.argv[1]
    results = combine_results(experiment_path)
    for model, result in results.items():
        write_csv(result, experiment_path, file_name=model+'_results.csv')