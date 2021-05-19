# Register data spaces
def get_data_space(data_space, debug):
    if debug:
        from physopt.data.debug_space import SPACE as DEBUG_SPACE
        return DEBUG_SPACE
    if data_space == 'TDW': # TODO: combine TDW and DEBUG space - i.e. get_tdw_data_space(debug)
        from physopt.data.tdw_space import SPACE as TDW_SPACE
        return TDW_SPACE
    elif data_space == "PHYSION":
        from physopt.data.physion_space import SPACE as PHYSION_SPACE
        return PHYSION_SPACE
    elif data_space == "DEBUG":
        from physopt.data.debug_space import SPACE as DEBUG_SPACE
        return DEBUG_SPACE
    else:
        raise ValueError('Unknown data: {0}'.format(data_space))
