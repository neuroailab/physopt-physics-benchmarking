# Register data spaces
from data_space.tdw_space import SPACE as TDW_SPACE
from data_space.debug_space import SPACE as DEBUG_SPACE



def get_data_space(data_space):
    if data_space == 'TDW':
        return TDW_SPACE
    elif data_space == 'DEBUG':
        return DEBUG_SPACE
    else:
        raise ValueError('Unknown data: {0}'.format(data_space))
