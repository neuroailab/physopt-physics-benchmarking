# Register data spaces
from physopt.data.tdw_space import SPACE as TDW_SPACE
from physopt.data.physion_space import SPACE as PHYSION_SPACE
from physopt.data.debug_space import SPACE as DEBUG_SPACE

def get_data_space(data_space, debug):
    if debug:
        return DEBUG_SPACE
    if data_space == 'TDW': # TODO: combine TDW and DEBUG space - i.e. get_tdw_data_space(debug)
        return TDW_SPACE
    elif data_space == "PHYSION":
        return PHYSION_SPACE
    elif data_space == "DEBUG":
        return DEBUG_SPACE
    else:
        raise ValueError('Unknown data: {0}'.format(data_space))
