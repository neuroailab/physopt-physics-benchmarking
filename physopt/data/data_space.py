import os

# Each subset is defined as a dict('name': NAME, 'data': DATA), where
# NAME = str, DATA = list(subset_paths)

EMPTY_DATA = [{'name': '', 'data': []}]


# All subsets combined
def get_combined_subset(subsets):
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
