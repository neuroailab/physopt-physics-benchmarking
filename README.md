# physopt

![](overview-figure.png)

## Overview

The goal of this repository is to train and evaluate different physics prediction models on one or many different physics scenarios. The inputs are model specific, however all currently implemented models predict from images. The output metrics are dataset specific, however all currently used datasets are evaluated on some form of binary prediction task. The procedure consists of two phases, as follows:

1. **Dynamics**: Train the physics prediction model on its specific prediction task on the specific train dataset.
  - Output: Model-specific trained checkpoint file saved to ``[OUTPUT_DIR]/[DYNAMICS_SCENARIO]/[SEED]/model/model.pt``
2. **Readout**: Evaulate the trained models under different readout protocols.
  1. __Feature Extraction__: Extract latent model features on the readout training and testing datasets.
    - Output: List of dicts, each dict corresponding to results from a batch, saved to `[OUTPUT_DIR]/[DYNAMICS_SCENARIO]/[SEED]/model/features/[READOUT_SCENARIO]/{train/test}_feat.pkl`
      - Each dict has the following keys: `input_states`, `observed_states`, `simulated_states`, `labels`, `stimulus_name`
  2. __Metrics Computation__: Train a classifier / regressor to predict the task using extracted latent train features and ground truth train labels, and test the trained classifier on the extracted latent test features to predict the test labels and evaluate them against the ground truth test labels using the the dataset specific test metric.
    - Output: Metric results and other data used for model-human comparisons saved to `[OUTPUT_DIR]/[DYNAMICS_SCENARIO]/[SEED]/model/features/[READOUT_SCENARIO]/metric_results.csv`
      - Used for analysis in [physics-benchmarking-neurips2021](https://github.com/cogtoolslab/physics-benchmarking-neurips2021)
     

## How to Install

Run `pip install -e .`


## How To Run

Physopt uses [Hyperopt](https://github.com/neuroailab/hyperopt) to train and evaluate physics prediction models on one or many different datasets. In order to evaluate a model, you first need to launch a MongoDB database, then a optimization server that distributes jobs across workers, and finally as many workers as you want that execute the server jobs.

For example, to run C-SWM, use:

`python opt.py --data physion --model CSWM --host localhost --port 25555 --database physion --output [OUTPUT_DIR]`

and then start up as many workers as you want with

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/physion --logfile=logfile.txt`

This approach has the advantage that you only need one set of workers pointing all to the same database `database`. Although currently not a problem, potential library conflicts between models might make approach c) infeasible in the future without separate python environments. In that case each model would have to be run in model-specific python environment which is beyond the scope of this documentation.

To see all available argument options use

`python opt.py --help`

## Dataset Specification

An overview of the required data settings and defaults can be found in `physopt/data/config.py`. 
- `DYNAMICS_TRAIN_DIR`, `DYNAMICS_TEST_DIR`, `READOUT_TRAIN_DIR`, and `READOUT_TEST_DIR` correspond to the base paths to the {dynamics/readout} {train/test} datasets.
- `SCENARIOS` is a list of the subdirs in each of the base dirs, which correspond to different datasets.
- `FILE_PATTERN` is used to allow for specifying what to match for to get each data file. 
- The fully constructed path provided to the dataloader will be `[BASE_DIR]/[SCENARIO]/[FILE_PATTERN]` (e.g. `../dynamics_train_data/Dominoes/*.hdf5`), which can be used by `glob` or similar to get the list of data files.

The default settings can be overwritten by using a `.yaml` file located in `physopt/data` and passing the filename (without the `.yaml` extension) as the `--data` commandline arg when running `opt.py`. See `physopt/data/physion.yaml` for an exmaple. 

`get_data_space`, defined in `physopt/data/data_space.py`, returns a list of dicts with the following structure:
-  `seed`: random seed to used initialize random generators
- `dynamics`: dict with `name`, `train`, and `test` that specify the dataset/scenario name, train datapaths, and test datapaths, respectively
- `readout`: same as for `dynamics` but for the readout phase instead

## Model Specification

An overview over all available models can be found in `physopt/models/__init__.py`.

To add a new model simply create a new `Objective` and update `physopt/models/__init__.py`. `Objective` inherits from `PhysOptObjective` as implemented in `physopt/utils.py` which primarily takes care of running the different phases and managing where intermediate results are stored. 

For a new `Objective` you will need to implement:
- `get_model`: Returns the model object
- `load_model`: Implements loading of model if model checkpoint file exists
- `save_model`: Implements saving of the model
- `get_dataloader`: Takes as input params `datapaths`, `phase`, `train`, and `shuffle`. Returns the dataloader object that can be iterated over for batches of data
- `train_step`: Takes as input a batch of data, performs the train optimization step, and returns the scalar loss value for that step
- `val_step`: Takes as input a batch of data, performs validation on that batch, and returns the scalar metric used for validation
- `extract_feat_step`: Takes as input a batc
- Optionally, `get_config`: Loads a configuration object. Must contain at least the settings in `physopt/models/config.py` and be accessible with dot notation. 
