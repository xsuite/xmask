def attach_beam_to_sequences(sequence, beam_to_configure=1, beam_configuration=None):
    """Attach beam to sequence

    Parameters
    ----------
    mad : pymadx.madx.Madx
        Madx object
    beam_to_configure : int
        Beam number to configure
    configuration : dict
        Beam configuration

    Returns
    -------
    None

    """
    mad = sequence._madx

    # beam energy
    mad.globals.nrj = beam_configuration['beam_energy_tot']
    particle_type = 'proton'

    if 'particle_mass' in beam_configuration.keys():
        particle_mass = beam_configuration['particle_mass']
        particle_type = 'ion'
    else:
        particle_mass = mad.globals.pmass # proton mass

    if 'particle_charge' in beam_configuration.keys():
        particle_charge = beam_configuration['particle_charge']
        particle_type = 'ion'
    else:
        particle_charge = 1.

    gamma_rel = (particle_charge*beam_configuration['beam_energy_tot'])/particle_mass
    # bv and bv_aux flags
    if beam_to_configure == 1:
        ss_beam_bv, ss_bv_aux = 1, 1
    elif beam_to_configure == 2:
        ss_beam_bv, ss_bv_aux = -1, 1
    elif beam_to_configure == 4:
        ss_beam_bv, ss_bv_aux = 1, -1
    else:
        raise ValueError("beam_to_configure must be 1, 2 or 4")

    mad.globals['bv_aux'] = ss_bv_aux
    mad.input(f'''
    beam, particle={particle_type},sequence={sequence.name},
        energy={beam_configuration['beam_energy_tot']*particle_charge},
        sigt={beam_configuration.get('beam_sigt', 0.0001)},
        bv={ss_beam_bv},
        npart={beam_configuration.get('beam_npart', 1)},
        sige={beam_configuration.get('beam_sige', 1e-6)},
        ex={beam_configuration.get('beam_norm_emit_x', 1) * 1e-6 / gamma_rel},
        ey={beam_configuration.get('beam_norm_emit_y', 1) * 1e-6 / gamma_rel},
        mass={particle_mass},
        charge={particle_charge};
    ''')

def configure_b4_from_b2(sequence_b4, sequence_b2,
        update_globals={'bv_aux': -1, 'mylhcbeam': 4}):

    mad_b2 = sequence_b2._madx
    mad_b4 = sequence_b4._madx

    var_dicts_b2 = mad_b2.get_variables_dicts()
    var_dicts_b4 = mad_b4.get_variables_dicts()

    b2_const=var_dicts_b2['constants']
    b4_const=var_dicts_b4['constants']
    for nn in b2_const.keys():
        if nn[0]=='_':
            print(f'The constant {nn} cannot be assigned!')
        else:
            if nn not in b4_const.keys():
                mad_b4.input(f'const {nn}={b2_const[nn]:.50e}')

    b2_indep=var_dicts_b2['independent_variables']
    b4_indep=var_dicts_b4['independent_variables']
    for nn in b2_indep.keys():
        mad_b4.input(f'{nn}={b2_indep[nn]:.50e}')

    b2_dep=var_dicts_b2['dependent_variables_expr']
    b4_dep=var_dicts_b4['dependent_variables_expr']
    for nn in b2_dep.keys():
        mad_b4.input(f'{nn}:={str(b2_dep[nn])}')

    # bv_aux and my my lhcbeam need to be defined explicitly
    for nn in update_globals.keys():
        mad_b4.input(f'{nn}={update_globals[nn]}')

    # Attach beam
    mad_b4.use(sequence_b2)
    beam_command = str(sequence_b2.beam)
    assert(', bv=-1.0' in beam_command)
    eam_command = beam_command.replace(', bv=-1.0', ', bv=1.0')
    mad_b4.input(beam_command)
    mad_b4.use(sequence_b4)

    # CHECKS
    var_dicts_b2 = mad_b2.get_variables_dicts()
    var_dicts_b4 = mad_b4.get_variables_dicts()

    b2_const=var_dicts_b2['constants']
    b4_const=var_dicts_b4['constants']
    for nn in b4_const.keys():
        assert b2_const[nn] == b4_const[nn]

    for nn in b2_const.keys():
        if nn not in b4_const.keys():
            print(f'Warning: b2 const {nn}={b2_const[nn]} is not in b4.')

    b2_indep=var_dicts_b2['independent_variables']
    b4_indep=var_dicts_b4['independent_variables']
    for nn in b2_indep.keys():
        if str(nn) in list(update_globals.keys()):
            continue
        assert b4_indep[nn] == b2_indep[nn]

    for nn in b4_indep.keys():
        if nn not in b2_indep.keys():
            print(f'Warning: b4 indep {nn}={b4_indep[nn]} is not in b2.')

    b2_dep=var_dicts_b2['dependent_variables_expr']
    b4_dep=var_dicts_b4['dependent_variables_expr']
    for nn in b2_dep.keys():
        if str(nn) in list(update_globals.keys()):
            continue
        assert str(b4_dep[nn]) == str(b2_dep[nn])

    for nn in b4_dep.keys():
        if nn not in b2_dep.keys():
            print(f'Warning: b4 dep {nn}={str(b4_dep[nn])} is not in b2.')