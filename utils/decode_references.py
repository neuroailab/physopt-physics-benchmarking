import sys
import os
import csv
import numpy as np



def decode_references(references_path):
    assert os.path.isfile(references_path) and references_path.endswith('.txt'), references_path

    with open(references_path, 'r') as f:
        lines = f.read().splitlines()
    
    refs = {}
    for l in lines:
        idx, path = l.split('->')
        idx = int(idx.replace('.hdf5', ''))
        if idx not in refs:
            refs[idx] = path
        else:
            raise KeyError('Index already exists in references! %d: %s' % (idx, path))

    return refs


if __name__ == '__main__':
    #references_path = '/mnt/fs1/tdw_datasets/pilot-clothSagging-redyellow/tfrecords/references.txt'
    references_path = sys.argv[1]
    refs = decode_references(references_path)
    for k, v in refs.items():
        print('%d: %s' % (k, v))
