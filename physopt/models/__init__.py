import importlib

def get_Objective(model): # TODO: figure out better implementation for this
    if model in {
        'pVGG_ID', 'pVGG_MLP', 'pVGG_LSTM',
        'pDEIT_ID', 'pDEIT_MLP', 'pDEIT_LSTM',
        'pCLIP_ID', 'pCLIP_MLP', 'pCLIP_LSTM',
        'pDINO_ID', 'pDINO_MLP', 'pDINO_LSTM',
        }:
        mymodule = importlib.import_module('physion.FROZEN')
        return getattr(mymodule, model + 'Objective')
    elif model == 'CSWM':
        from cswm.CSWM import Objective
        return Objective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
