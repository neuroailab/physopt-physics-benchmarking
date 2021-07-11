# Register data spaces
def get_data_space(data_space, debug):
    if data_space == 'DEBUG':
        from physopt.data.debug_space import SPACE as DEBUG_SPACE
        return DEBUG_SPACE
    elif data_space == 'TDW': # TODO: combine TDW and DEBUG space - i.e. get_tdw_data_space(debug)
        from physopt.data.tdw_space import SPACE as TDW_SPACE
        return TDW_SPACE
    elif data_space == 'DOMINOES':
        from physopt.data.dominoes_space import SPACE as DOMINOES_SPACE
        return DOMINOES_SPACE
    elif data_space == 'PHYSION':
        from physopt.data.physion_space import SPACE as PHYSION_SPACE
        return PHYSION_SPACE
    elif data_space == 'PHYSION_TF': 
        from physopt.data.physion_space_tf import SPACE as PHYSION_SPACE_TF
        return PHYSION_SPACE_TF
    else:
        raise ValueError('Unknown data: {0}'.format(data_space))
