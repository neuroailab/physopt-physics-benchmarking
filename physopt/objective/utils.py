import os
import logging
import errno
import traceback
import time
import collections
import mlflow
import psycopg2
import boto3
import botocore

PRETRAINING_PHASE_NAME = 'pretraining'
EXTRACTION_PHASE_NAME = 'extraction'
READOUT_PHASE_NAME = 'readout'

def flatten(d, parent_key='', sep='_', prefix=None):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    if prefix is not None:
        items = [(prefix+sep+k, v) for k,v in items]
    return dict(items)

def setup_logger(log_file, debug=False):
    _create_dir(log_file)
    logging.root.handlers = [] # necessary to get handler to work
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
            ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        )

def get_output_dir(output_dir, experiment_name, model_name, pretraining_name, seed, phase, readout_name):
    assert pretraining_name is not None
    if readout_name is None:
        model_dir = os.path.join(output_dir, experiment_name, model_name, pretraining_name, str(seed), phase, '')
    else:
        model_dir = os.path.join(output_dir, experiment_name, model_name, pretraining_name, str(seed), phase, readout_name, '')
    assert model_dir[-1] == '/', '"{}" missing trailing "/"'.format(model_dir) # need trailing '/' to make dir explicit
    _create_dir(model_dir)
    return model_dir

def _create_dir(path): # creates dir from path or filename, if doesn't exist
    dirname, basename = os.path.split(path)
    assert '.' in basename if basename else True, 'Are you sure filename is "{}", or should it be a dir'.format(basename) # checks that basename has file-extension
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def create_postgres_db(host, port, dbname):
    connection = None
    try:
        connection = psycopg2.connect(f"user='physopt' password='physopt' host='{host}' port='{port}' dbname='postgres'") # use postgres db just for connection
        logging.info('Database connected.')
    except Exception as e:
        logging.info('Database not connected.')
        raise e

    if connection is not None:
        connection.autocommit = True
        cur = connection.cursor()
        cur.execute("SELECT datname FROM pg_database;")
        list_database = cur.fetchall()

        if (dbname,) in list_database:
            logging.info(f"'{dbname}' Database already exist")
        else:
            logging.info(f"'{dbname}' Database not exist.")
            sql_create_database = f'create database "{dbname}";'
            cur.execute(sql_create_database)
        connection.close()

def get_run_name(model_name, pretraining_name, seed, phase, readout_name=None, separator='-', **kwargs): # TODO: remove need for unused kwargs
    to_join = [model_name, pretraining_name, str(seed), phase]
    if readout_name is not None:
        to_join.append(readout_name)
    return separator.join(to_join)

def get_exp_name(name, add_ts=False, debug=False):
        if debug:
            experiment_name = 'DEBUG'
        elif add_ts:
            experiment_name = name + '_' + time.strftime("%Y%m%d-%H%M%S")
        else:
            experiment_name = name
        return experiment_name

def get_mlflow_backend(output_dir, host, port, dbname): # TODO: split this?
    if dbname  == 'local':
        tracking_uri = os.path.join(output_dir, 'mlruns')
        artifact_location = None
    else:
        # create postgres db, and use for backend store
        create_postgres_db(host, port, dbname)
        tracking_uri = 'postgresql://physopt:physopt@{}:{}/{}'.format(host, port, dbname) 

        # create s3 bucket, and use for artifact store
        s3 = boto3.resource('s3')
        s3.create_bucket(Bucket=dbname)
        artifact_location =  's3://{}'.format(dbname) # TODO: add run name to make it more human-readable?
    return tracking_uri, artifact_location

def download_from_artifact_store(artifact_path, tracking_uri, run_id, output_dir): # Tries to download artifact, returns None if not found
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        client.download_artifacts(run_id, artifact_path, output_dir)
        logging.info(f'Downloaded {artifact_path} to {output_dir}')
        output_file = os.path.join(output_dir, artifact_path)
    except (FileNotFoundError, botocore.exceptions.ClientError):
        logging.info(f"Couldn't find artifact at {artifact_path} in artifact store")
        logging.debug(traceback.format_exc())
        output_file = None
    return output_file

def get_ckpt_from_artifact_store(step, tracking_uri, run_id, output_dir): # returns path to downloaded ckpt, if found
    artifact_path = f'step_{step}/model_ckpts/model.pt'
    model_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, output_dir)
    return model_file

def get_feats_from_artifact_store(mode, step, tracking_uri, run_id, output_dir): # returns path to downloaded feats, if found
    artifact_path = f'step_{step}/features/{mode}_feat.pkl'
    feat_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, output_dir)
    return feat_file

def get_readout_model_from_artifact_store(protocol, step, tracking_uri, run_id, output_dir): # returns path to downloaded readout model, if found
    artifact_path = f'step_{step}/readout_models/{protocol}_readout_model.joblib'
    readout_model_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, output_dir)
    return readout_model_file

def get_run(tracking_uri, experiment_id, **kwargs):
    logging.info(f'Searching for run with params: {kwargs}')
    runs = search_runs(tracking_uri, experiment_id, **kwargs)
    if 'run_name' in kwargs:
        run_name = kwargs['run_name']
    else:
        run_name = get_run_name(**kwargs)
    assert len(runs) <= 1, f'Should be at most one (1) run with name "{run_name}", but found {len(runs)}'
    if len(runs) == 0:
        logging.info(f'Creating run with name:"{run_name}"')
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run = client.create_run(experiment_id, tags={'mlflow.runName': run_name})
    else: # found existing run with matching name
        run = runs[0]
        logging.info(f'Found run with name: "{run.data.tags["mlflow.runName"]}"')
    return run

def search_runs(tracking_uri, experiment_id, **kwargs):
    filters = []
    for param, value in kwargs.items():
        if param == 'run_name':
            filters.append(f'tags.mlflow.runName="{value}"')
        else:
            filters.append(f'params.{param}="{value}"')
    filter_string = ' and '.join(filters)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs([experiment_id], filter_string=filter_string)
    return runs

def create_experiment(tracking_uri, experiment_name, artifact_location):
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None: # create experiment if doesn't exist
        logging.info('Creating new experiment with name "{}"'.format(experiment_name))
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
        experiment = client.get_experiment(experiment_id)
    else: # uses old experiment settings (e.g. artifact store location)
        logging.info('Experiment with name "{}" already exists'.format(experiment_name))
        # TODO: check that experiment settings match?
    return experiment

