def get_Objective(model): # TODO: figure out better implementation for this
    if model == 'RPIN':
        from physopt.models.RPIN import Objective as RPINObjective
        return RPINObjective
    elif model == 'SVG':
        from physopt.models.SVG import VGGObjective as SVGObjective
        return SVGObjective
    elif model == 'SVGPretrained':
        from physopt.models.SVG import VGGPretrainedObjective as SVGPretrainedObjective
        return SVGPretrainedObjective
    elif model == 'SVGPretrainedFrozen':
        from physopt.models.SVG import VGGPretrainedFrozenObjective as SVGPretrainedFrozenObjective
        return SVGPretrainedFrozenObjective
    elif model == 'SVGPretrainedDEIT':
        from physopt.models.SVG import DEITPretrainedObjective as SVGPretrainedDEITObjective
        return SVGPretrainedDEITObjective
    elif model == 'SVGPretrainedFrozenDEIT':
        from physopt.models.SVG import DEITPretrainedFrozenObjective as SVGPretrainedFrozenDEITObjective
        return SVGPretrainedFrozenDEITObjective
    elif model == 'SVGPretrainedCLIP':
        from physopt.models.SVG import CLIPPretrainedObjective as SVGPretrainedCLIPObjective
        return SVGPretrainedCLIPObjective
    elif model == 'SVGPretrainedFrozenCLIP':
        from physopt.models.SVG import CLIPPretrainedFrozenObjective as SVGPretrainedFrozenCLIPObjective
        return SVGPretrainedFrozenCLIPObjective
    elif model == 'VGGFrozenID':
        from  physopt.models.physion.FROZEN import VGGFrozenIDObjective
        return VGGFrozenIDObjective
    elif model == 'VGGFrozenMLP':
        from  physopt.models.physion.FROZEN import VGGFrozenMLPObjective
        return VGGFrozenMLPObjective
    elif model == 'VGGFrozenLSTM':
        from  physopt.models.physion.FROZEN import VGGFrozenLSTMObjective
        return VGGFrozenLSTMObjective
    elif model == 'DEITFrozenID':
        from  physopt.models.physion.FROZEN import DEITFrozenIDObjective
        return DEITFrozenIDObjective
    elif model == 'DEITFrozenMLP':
        from  physopt.models.physion.FROZEN import DEITFrozenMLPObjective
        return DEITFrozenMLPObjective
    elif model == 'DEITFrozenLSTM':
        from  physopt.models.physion.FROZEN import DEITFrozenLSTMObjective
        return DEITFrozenLSTMObjective
    elif model == 'CLIPFrozenID':
        from  physopt.models.physion.FROZEN import CLIPFrozenIDObjective
        return CLIPFrozenIDObjective
    elif model == 'CLIPFrozenMLP':
        from  physopt.models.physion.FROZEN import CLIPFrozenMLPObjective
        return CLIPFrozenMLPObjective
    elif model == 'CLIPFrozenLSTM':
        from  physopt.models.physion.FROZEN import CLIPFrozenLSTMObjective
        return CLIPFrozenLSTMObjective
    elif model == 'DINOFrozenID':
        from  physopt.models.physion.FROZEN import DINOFrozenIDObjective
        return DINOFrozenIDObjective
    elif model == 'DINOFrozenMLP':
        from  physopt.models.physion.FROZEN import DINOFrozenMLPObjective
        return DINOFrozenMLPObjective
    elif model == 'DINOFrozenLSTM':
        from  physopt.models.physion.FROZEN import DINOFrozenLSTMObjective
        return DINOFrozenLSTMObjective
    elif model == 'CSWM':
        from physopt.models.cswm.CSWM import Objective
        return Objective
    elif model == 'OP3':
        from physopt.models.op3.OP3 import Objective
        return Objective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
