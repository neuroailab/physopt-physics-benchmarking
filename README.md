# physopt

![](overview-figure.png)

## Overview

The goal of this repository is to train and evaluate different physics prediction models under various pretraining and readout protocols. The procedure consists of three phases, as follows:

1. **Pretraining**: Train the physics prediction model on its specific prediction task on the specific train dataset.
    - Output: Model-specific trained checkpoint file saved to ``[OUTPUT_DIR]/[PRETRAINING_SCENARIO]/[SEED]/model/model.pt``
2. **Extraction**: Extract model features for the readout training and testing datasets.
    - Output: List of dicts, each dict corresponding to results from a batch, saved to `[OUTPUT_DIR]/[PRETRAINING_SCENARIO]/[SEED]/model/features/[READOUT_SCENARIO]/{train/test}_feat.pkl`
    - Each dict has the following keys: `input_states`, `observed_states`, `simulated_states`, `labels`, `stimulus_name`
3. **Readout**: Train a model to predict the task labels using extracted features, and evaluate the trained readout model on the readout test set.
    - Output: Metric results and other data used for model-human comparisons saved to `[OUTPUT_DIR]/[PRETRAINING_SCENARIO]/[SEED]/model/features/[READOUT_SCENARIO]/metric_results.csv`
    - Used for analysis in [physics-benchmarking-neurips2021](https://github.com/cogtoolslab/physics-benchmarking-neurips2021)

Runs and artifacts from running the pipeline are recorded with [MLflow](https://mlflow.org/).

## How to Install
**Recommended**: Create a virtualenv with `virtualenv -p python3 .venv` and activate it using `source .venv/bin/activate`. Note that you will likely run into issues if you use python3 if it is an older version. python3.5 may cause issues while python3.7 appears to be fine. You may be able to find later python versions in /usr/bin.

Run `pip install -e .` in the root `physopt` directory to install the `physopt` package. You will also need to install the correct version of PyTorch for your system, see [this link](https://pytorch.org/get-started/locally/) for instructions.

In order to use PostgreSQL as the MLflow backend store, you'll need to install postgresql with `sudo apt-get install postgresql`, if it's not installed already -- you can check with `psql --version`.

## How To Run

### Local
To run, use `run` from `physopt.opt`. The only required commandline argument is  `--config` or  `-C`. Optionally, you may also choose to specifiy the output directory where the results are saved (with `--output` or `-O`). The mlflow backend store is set to `[OUTPUT_DIR]/mlruns`.

For convenience, you can use the following environment variables: `PHYSOPT_CONFIG_DIR`, which specifies the directory to look for configs if passed a relative path, and `PHYSOPT_OUTPUT_DIR`, which specifies the output directory to use if none is specified in the commandline. Also, `setup_environment_vars` is also provided in `physopt.opt` which allows you to specify the environment variables using a `.yaml` file.

### Remote MLflow Tracking Server
MLflow allows for using a remote Tracking Server. Specifically, we use a Postgres database for backend entity storage and an S3 bucket for artifact storage. This requires setting up PostgreSQL and Amazon S3 as detailed in the [Setup](#setup) section above. The relevant commandline arguments are the port (`--postgres_port`) and database name (`--postgres_dbname`). Note that the name "local" is reserved for using local storage. Also if the Postgres server is not running on `localhost` you'll need to specify the host (`--postgres_host`). 

Therefore, the command would look like, 

`python opt.py --data_module [DATA_SPACE_MODULE_NAME] --objective_module [OBJECTIVE_MODULE_NAME] --postgres_port [PORT] --postgres_dbname [DBNAME] (--postgres_host [HOST]) (--ouput [OUTPUT_DIR]) (--num_threads [NUM_THREADS])`.

## Configuration 
The default configuration can be found in `physopt/config.py`, which is updated by specifying a YAML configuration file using the `--config` (or `-C`) commandline argument. The following are required:
- `DATA_SPACE.MODULE` (see [data spaces specification](#data-spaces-specification))
- `PRETRAINING.OBJECTIVE_MODULE` (see [model specification](#model-specification))
- `PRETRAINING.MODEL_NAME` 
- `EXTRACTION.OBJECTIVE_MODULE` (see [model specification](#model-specification))

### Data Spaces Specification
The `DATA_SPACE.FUNC` (defaults to `get_data_spaces`) from the specified `DATA_SPACE.MODULE` must return a list of dicts with the following structure:
- `pretraining`: dict with `name`, `train`, and `test` that specify the dataset/scenario name, train datapaths, and test datapaths, respectively
- `readout`: a list of dicts, with each dict having the same format as in `pretraining` but specifying data for readout phase instead

Any `kwargs` for `DATA_SPACE.FUNC` can be specified using `DATA_SPACE.KWARGS`.

The seeds, specified by `DATA_SPACE.SEEDS`, should  be a list  of seeds to use. Each set of pretraining and readout datasets (i.e. each element of the list of dicts returned by `DATA_SPACE.FUNC`) will be run with each seed.

An example of how the data spaces can be constructed can be found in the [Physion](https://github.com/neuroailab/physion/tree/master/physion/data_space) repo.

###  Model Specification
Running a model in `physopt` requires creating an Objective class for each phase (pretraining, extraction, and readout), specified by `[PHASE]_OBJECTIVE.MODULE` and `[PHASE]_OBJECTIVE.NAME`. 

Your `PretrainingObjective` should inherit from `PretrainingObjectiveBase` ([link](https://github.com/neuroailab/physopt/blob/0801ba64506edebe0a56f1a16948d8d42fc7fea3/physopt/objective/base.py#L17)) and requires implmenting the following methods:
- `get_pretraining_dataloader`: Takes as input params a list of `datapaths` and a bool `train` flag. Returns the dataloader object that can be iterated over for batches of data
- `train_step`: Takes as input a batch of data, performs the train optimization step, and returns the scalar loss value for that step
- `val_step`: Takes as input a batch of data, performs validation on that batch, and returns the scalar metric used for validation

Your `ExtractionObjective` should inherit from `ExtractionObjecitveBase` and requires implmenting the following methods:
- `get_readout_dataloader`: Takes as input params a list of `datapaths`. Returns the dataloader object that can be iterated over for batches of data
- `extract_feat_step`: Takes as input a batch of data, and outputs a dict with `input_states`, `observed_states`, `simulated_states`, `labels`, and `stimulus_name`

A simple logistic regression readout model is provided, but a different `ReadoutObjective` can be used by inheriting from `ReadoutObjectiveBase` and implementing:
- `get_readout_model`: Returns a model object that has the following methods: `fit`, `predict`, and `predict_proba`.

The `PretrainingObjective` and `ExtractionObjective` both also inherit from `PhysOptModel`, which requires implementing:
- `get_model`: Returns the model object
- `load_model`: Implements loading of the model given a model checkpoint file
- `save_model`: Implements saving of the model given a model checkpoint file

An example can be found [here](https://github.com/neuroailab/physion/blob/fe10826dffef59bd866f388202b6dadc5b3f91d4/physion/models/frozen.py).

## Setup
### PostgreSQL
Connect to the PostgreSQL server using `sudo -u postgres psql`. You should see the prompt start with `postgres=#`. Next, create a user with username and password "physopt" using `CREATE USER physopt WITH PASSWORD 'physopt' CREATEDB;`. Verify that the user was created successfully with `\du`. 

You can change the port by changing the setting in the `postgresql.conf` file, whose location can be shown using `SHOW config_file;`. After you change `postgresql.conf` make sure to restart the server using `sudo service postgresql restart`. You can check what port is being used with `\conninfo` after connecting to the server.

### Amazon S3
In order to use S3 as the MLflow artifact store, you'll need to add your AWS credentials to `~/.aws/credentials`. See [this link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) for more information about the AWS credential file.


### MLflow Tracking UI
To view the MLflow tracking UI run `mlflow ui`. If you are using local storage add `--backend-store-uri file:///[OUTPUT_DIR]/mlruns`. Otherwise, if you're using the PostgreSQL backend add `--backend-store-uri postgresql://<username>:<password>@<host>:<port>/<database>`. Finally, navigate to `http://localhost:5000`.

#### Notes
If the machine running the PostgreSQL and MLflow tracking servers is not publically visible, you'll need to setup the necessary ssh tunnels.

## Citing Physion

If you find this codebase useful in your research, please consider citing:

    @inproceedings{bear2021physion,
        Title={Physion: Evaluating Physical Prediction from Vision in Humans and Machines},
        author= {Daniel M. Bear and
               Elias Wang and
               Damian Mrowca and
               Felix J. Binder and
               Hsiao{-}Yu Fish Tung and
               R. T. Pramod and
               Cameron Holdaway and
               Sirui Tao and
               Kevin A. Smith and
               Fan{-}Yun Sun and
               Li Fei{-}Fei and
               Nancy Kanwisher and
               Joshua B. Tenenbaum and
               Daniel L. K. Yamins and
               Judith E. Fan},
        url = {https://arxiv.org/abs/2106.08261},
        archivePrefix = {arXiv},
        eprint = {2106.08261},
        Year = {2021}
    }
