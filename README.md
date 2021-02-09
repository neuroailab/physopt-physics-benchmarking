# physopt

## Overview
The goal of this repository is to train and evaluate different physics prediction models on one or many different physics scenarios. The inputs are model specific, however all currently implemented models predict from images. The output metrics are dataset specific, however all currently used datasets are evaluated on some form of binary prediction task. The procedure to extract metrics for each dataset and model is as follows:

1. Train the physics prediction model on its specific prediction task on the specific train dataset.
  - Input: Model-specific train data (e.g. images and bounding boxes).
  - Output: Model-specific trained checkpoint file
    - stored at `{output_directory}/{train_data_name}/{seed}/model`
    - stored as `model-specific (e.g. pytorch or tensorflow checkpoint file)`
2. Extract latent model features on the specific train dataset.
  - Input: Model-specific train data and labels and trained checkpoint file.
  - Output: Latent train features and labels per data example
    - stored at `{output_directory}/{train_data_name}/{seed}/model/features/{train_feat_data_name}/feat.pkl`
    - stored as `list(batch) with batch = dict(encoded_states: visual_features, rollout_states: predicted_features, binary_labels: labels]))`.
3. Extract latent model features on the specific test dataset.
  - Input: Model-specific test data and labels and trained checkpoint file.
  - Output: Latent test features and labels per data example 
    - stored at `{output_directory}/{train_data_name}/{seed}/model/features/{test_feat_data_name}/feat.pkl`
    - stored as `list(batch) with batch = dict(encoded_states: visual_features, rollout_states: predicted_features, binary_labels: labels))`.
4. Train a classifier / regressor to predict the task using extracted latent train features and ground truth train labels, and test the trained classifier on the extracted latent test features to predict the test labels and evaluate them against the ground truth test labels using the the dataset specific test metric.
  - Input: Latent train and test features and labels.
  - Output: Test dataset specific metrics 
    - stored at `{output_directory}/{train_data_name}/{seed}/model/features/{test_feat_data_name}/metrics_results.pkl`
    - stored as `list(setting_result) with setting_result = dict(results: metric_results, seed: seed, train_name: train_model_data_name, train_feat_name: train_feat_data_name, test_feat_name: test_feat_data_name, model_dir: model_dir)`. 

## How to Install

Run

`pip install -r requirements.txt`

and add [RPIN](https://github.com/neuroailab/RPIN), [SVG](https://github.com/neuroailab/svg), [CLIP](https://github.com/openai/CLIP) and [physopt](https://github.com/neuroailab/physopt) to your PYTHONPATH with 

`export PYTHONPATH=$PYTHONPATH:{path_to_RPIN}:{path_to_SVG}:{path_to_CLIP}:{path_to_physopt}`.

## How To Run

Physopt uses [Hyperopt](https://github.com/neuroailab/hyperopt) to train and evaluate physics prediction models on one or many different datasets. In order to evaluate a model, you first need to launch a MongoDB database, then a optimization server that distributes jobs across workers, and finally as many workers as you want that execute the server jobs.

For example,

a) to evaluate ROI pooling run

`python opt.py --data TDW --model RPIN --host localhost --port 25555 --database rpin --output rpin_output_directory --num_threads 1`

and then start up as many workers as you want with

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/rpin --logfile=logfile.txt`


b) to evaluate SVG run

`python opt.py --data TDW --model SVG --host localhost --port 25555 --database svg --output svg_output_directory --num_threads 1`

and then start up as many workers as you want with:

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/svg --logfile=logfile.txt`

c) to evaluate multiple models run in separate threads:

`python opt.py --data TDW --model RPIN --host localhost --port 25555 --database database --output rpin_output_directory --num_threads 1`

`python opt.py --data TDW --model SVG --host localhost --port 25555 --database database --output svg_output_directory --num_threads 1`

etc.

and then start up as many workers as you want with:

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/database --logfile=logfile.txt`

This approach has the advantage that you only need one set of workers pointing all to the same database `database`. Although currently not a problem, potential library conflicts between models might make approach c) infeasible in the future without separate python environments. In that case each model would have to be run in model-specific python environment which is beyond the scope of this documentation.

To see all available argument options use

`python opt.py --help`

## Model Specification

An overview over all available models can be found in [physopt/models/\_\_init\_\_.py](https://github.com/neuroailab/physopt/blob/main/physopt/models/__init__.py).

## Dataset Specification

An overview over all available datasets can be found in [physopt/data/\_\_init\_\_.py](https://github.com/neuroailab/physopt/blob/main/physopt/data/__init__.py).

To add a new dataset simply follow the format outlined in [physopt/data/tdw\_space.py](https://github.com/neuroailab/physopt/blob/main/physopt/data/tdw_space.py) and add the newly created dataset to `get_data_space` in [physopt/data/\_\_init\_\_.py](https://github.com/neuroailab/physopt/blob/main/physopt/data/__init__.py).

Physopt first trains a selected model on all the datasets defined in `TRAIN_DATA` and seeds defined in `SEEDS`, then extracts the train features for the datasets defined in `TRAIN_FEAT_DATA` and test features for the datasets defined in `TEST_FEAT_DATA` and finally use the extracted features to calculate the evaluation metrics for the train-test dataset pairs defined in `METRICS_DATA`. A seed is an integer. Each `DATA` is a list of single datasets. A single dataset is a dictionary of `{'name': dataset_name, 'data': list(dataset_paths)}`. `METRICS_DATA` is a tuple of `(TRAIN_FEAT_DATA, TEST_FEAT_DATA)`.
