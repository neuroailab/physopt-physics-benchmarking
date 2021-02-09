# Register models
import physopt.metrics.physics.test_metrics as test_metrics
from physopt.models.RPIN import Objective as RPINObjective
from physopt.models.SVG import VGGObjective as SVGObjective


def get_Objective(model):
    if model == 'metrics':
        return test_metrics.Objective
    elif model == 'RPIN':
        return RPINObjective
    elif model == 'SVG':
        return SVGObjective
    else:
        raise ValueError('Unknown model: {0}'.format(model))
