import os

# Each subset is defined as a dict('name': NAME, 'data': DATA), where
# NAME = str, DATA = list(subset_paths)

EMPTY_DATA = [{'name': '', 'data': []}]


# All subsets combined
def get_combined_subset(subsets): # return list for consistency?
    return {'name': 'all', 'data': [s for subset in subsets for s in subset['data']]}


# All subsets combined but one
def get_combined_but_one_subsets(subsets):
    combined_but_one = []
    for subset in subsets:
        combined_data = get_combined_subset(subsets)['data']
        [combined_data.remove(s) for s in subset['data']]
        combined_but_one.append({'name': 'no_{0}'.format(subset['name']), 'data': combined_data})
    return combined_but_one


def get_all_subsets(subsets):
    if not isinstance(subsets, list):
        subsets = list(subsets)
    if len(subsets) > 1:
        all_subsets = subsets \
                + get_combined_but_one_subsets(subsets) \
                + [get_combined_subset(subsets)]
    else:
        all_subsets = subsets
    return all_subsets


def construct_extraction_space(seeds, train_data, feat_data):
    return (seeds, train_data, feat_data)


def construct_metrics_space(seeds, train_data, metrics_data):
    return (seeds, train_data, metrics_data)


def construct_data_spaces(seeds, train_data, train_feat_data, test_feat_data, metrics_data):
    spaces = {}
    spaces['train_feat'] = construct_extraction_space(seeds, train_data, train_feat_data)
    spaces['test_feat'] = construct_extraction_space(seeds, train_data, test_feat_data)
    spaces['metrics'] = construct_metrics_space(seeds, train_data, metrics_data)
    return spaces

def get_data_space(
        data_space,
        debug,
        train_dir='/data1/eliwang/physion_train_data', # TODO
        test_dir='/data1/eliwang/physion_test_data', # TODO
        scenarios=['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support'],
        num_seeds=1,
        ):

    # Data subsets
    train_data = []
    train_feat_data = []
    test_feat_data = []
    for scenario in scenarios:
        train_data.append({'name': scenario, 'data': [os.path.join(train_dir, 'dynamics_training', scenario, '*.hdf5')]}) # TODO: might need to make compatible with tfrecords too
        train_feat_data.append({'name': scenario, 'data': [os.path.join(train_dir, 'readout_training', scenario, '*.hdf5')]})
        test_feat_data.append({'name': scenario, 'data': [os.path.join(test_dir, 'model_testing', scenario, '*.hdf5')]})

    # Spaces
    seeds = list(range(num_seeds))

    train_data = get_all_subsets(train_data) # TODO: have param specifying what train protocol all/abo/only
    metrics_data = zip(train_feat_data, test_feat_data)

    space = construct_data_spaces(seeds, train_data, train_feat_data, test_feat_data, metrics_data)
    return space
