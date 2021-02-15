# Register data spaces
from physopt.data.tdw_space import SPACE as TDW_SPACE
from physopt.data.debug_space import SPACE as DEBUG_SPACE

def get_data_space(data_space, debug):
    if data_space == 'TDW': # TODO: combine TDW and DEBUG space - i.e. get_tdw_data_space(debug)
        if debug:
            return DEBUG_SPACE
        else:
            return TDW_SPACE
    else:
        raise ValueError('Unknown data: {0}'.format(data_space))
