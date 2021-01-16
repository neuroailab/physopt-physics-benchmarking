# physopt

Physopt uses [Hyperopt](https://github.com/neuroailab/hyperopt) to train and evaluate physics prediction models on one or many different datasets. In order to evaluate a model, you first need to launch a MongoDB database, then a optimization server that distributes jobs across workers, and finally as many workers as you want that execute the server jobs.

For example,

a) to evaluate ROI pooling run

`python opt.py --model RPIN --host localhost --port 25555 --database rpin --output rpin_output_directory --num_threads 1`

and then start up as many workers as you want with

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/rpin --logfile=logfile.txt`


b) to evaluate SVG run

`python opt.py --model SVG --host localhost --port 25555 --database svg --output svg_output_directory --num_threads 1`

and then start up as many workers as you want with:

`hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/svg --logfile=logfile.txt`

etc.

Physopt will then first train the selected model on all the datasets defined in `TRAIN_SPACE`, then extract the train features for the datasets defined in `TRAIN_FEAT_SPACE` and test features for the datasets defined in `HUMAN_FEAT_SPACE` and finally use the extracted features to calculate the evaluation metrics for the datasets defined in `METRICS_SPACE`. Each `SPACE` consists of a tuple `(SEEDS, TRAIN_DATA, FEAT_DATA)` which specify a list of possible seeds, train datasets, and feature datasets respectively. A seed is an integer. A dataset is a dictionary of `{'name': dataset\_name, 'data': list(dataset_paths)}`. The feature space `FEAT_DATA` of `METRICS_SPACE` is a tuple of `(TRAIN_FEAT_DATA, TEST_FEAT_DATA)`, otherwise a feature dataset `FEAT_DATA`.
