def get_Objective(model):
    if model == 'metrics':
        import physopt.metrics.physics.test_metrics as test_metrics
        return test_metrics.Objective
    elif model == 'RPIN':
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
    elif model == 'VGGFrozenMLP':
        from  physopt.models.physion.FROZEN import VGGFrozenMLPObjective
        return VGGFrozenMLPObjective
    elif model == 'VGGFrozenLSTM':
        from  physopt.models.physion.FROZEN import VGGFrozenLSTMObjective
        return VGGFrozenLSTMObjective
    elif model == 'DEITFrozenMLP':
        from  physopt.models.physion.FROZEN import DEITFrozenMLPObjective
        return DEITFrozenMLPObjective
    elif model == 'DEITFrozenLSTM':
        from  physopt.models.physion.FROZEN import DEITFrozenLSTMObjective
        return DEITFrozenLSTMObjective
    elif model == 'CSWM':
        from physopt.models.cswm.CSWM import Objective
        return Objective
    elif model == 'OP3':
        from physopt.models.op3.OP3 import Objective
        return Objective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
