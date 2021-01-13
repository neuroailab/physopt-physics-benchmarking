import os

# Each subset is defined as a dict('name': NAME, 'data': DATA), where
# NAME = str, DATA = list(subset_paths)

# All subsets combined
def get_combined_subset(subsets):
    return {'name': 'all', 'data': [s for subset in subsets for s in subset['data']]}


# All subsets combined but one
def get_combined_but_one_subsets(subsets):
    combined_but_one = []
    for subset in subsets:
        combined_data = get_combined_subset(subsets)['data']
        [combined_data.remove(s) for s in subset['data']]
        combined_but_one.append(('no_{0}'.format(subset['name']), combined_data))
    return combined_but_one


def get_all_subsets(subsets):
    all_subsets = subsets \
            + get_combined_but_one_subsets(subsets) \
            + [get_combined_subset(subsets)]
    return all_subsets
