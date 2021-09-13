import importlib

def get_Objective(model): # TODO: figure out better implementation for this
    if model == 'RPIN':
        from physopt.models.RPIN import Objective as RPINObjective
        return RPINObjective
    elif model == 'SVG':
        from physopt.models.SVG import VGGObjective as SVGObjective
        return SVGObjective
    elif model in {
        'pVGG_ID', 'pVGG_MLP', 'pVGG_LSTM',
        'pDEIT_ID', 'pDEIT_MLP', 'pDEIT_LSTM',
        'pCLIP_ID', 'pCLIP_MLP', 'pCLIP_LSTM',
        'pDINO_ID', 'pDINO_MLP', 'pDINO_LSTM',
        }:
        mymodule = importlib.import_module('physopt.models.physion.FROZEN')
        return getattr(mymodule, model + 'Objective')
    elif model == 'CSWM':
        from physopt.models.cswm.CSWM import Objective
        return Objective
    elif model == 'OP3':
        from physopt.models.op3.OP3 import Objective
        return Objective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
