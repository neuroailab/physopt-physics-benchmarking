# Register models
import physopt.metrics.physics.test_metrics as test_metrics
from physopt.models.RPIN import Objective as RPINObjective
from physopt.models.SVG import VGGObjective as SVGObjective
from physopt.models.SVG import VGGPretrainedObjective as SVGPretrainedObjective
from physopt.models.SVG import VGGPretrainedFrozenObjective as SVGPretrainedFrozenObjective
from physopt.models.SVG import DEITPretrainedObjective as SVGPretrainedDEITObjective
from physopt.models.SVG import DEITPretrainedFrozenObjective as SVGPretrainedFrozenDEITObjective
from physopt.models.SVG import CLIPPretrainedObjective as SVGPretrainedCLIPObjective
from physopt.models.SVG import CLIPPretrainedFrozenObjective as SVGPretrainedFrozenCLIPObjective


def get_Objective(model):
    if model == 'metrics':
        return test_metrics.Objective
    elif model == 'RPIN':
        return RPINObjective
    elif model == 'SVG':
        return SVGObjective
    elif model == 'SVGPretrained':
        return SVGPretrainedObjective
    elif model == 'SVGPretrainedFrozen':
        return SVGPretrainedFrozenObjective
    elif model == 'SVGPretrainedDEIT':
        return SVGPretrainedDEITObjective
    elif model == 'SVGPretrainedFrozenDEIT':
        return SVGPretrainedFrozenDEITObjective
    elif model == 'SVGPretrainedCLIP':
        return SVGPretrainedCLIPObjective
    elif model == 'SVGPretrainedFrozenCLIP':
        return SVGPretrainedFrozenCLIPObjective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
