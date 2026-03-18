import xtrack as xt
import numpy as np
from scipy.special import factorial

from lhc_geography import SIDE_APER_TO_SIDE_BEAM

PREFIX_TO_MAIN_ORDER = [
    ('mb', (0, 'normal')),
    ('mqs', (1, 'skew')), ('mq', (1, 'normal')),
    ('mss', (2, 'skew')), ('ms', (2, 'normal')),
    ('mo', (3, 'normal')),
    ('mcbh', (0, 'normal')), ('mcbv', (0, 'skew')),
    ('mcbch', (0, 'normal')), ('mcbcv', (0, 'skew')),
    ('mcssx', (2, 'skew')), ('mcsx', (2, 'normal')),
    ('mcs', (2, 'normal')),
    ('mco', (3, 'normal')),
    ('mcd', (4, 'normal')),
    ('mct', (5, 'normal')),
]

def load_wise_table_arc_magnets(fname_err_table, fname_rotations, min_order=2, max_order=15, ref_radius=0.017):
    # Load WISE error table
    tt_raw = xt.Table.from_tfs(fname_err_table)
    tt_err_data = tt_raw._data.copy()
    tt_err_data['name'] = np.array([nn.lower() for nn in tt_err_data['name']])
    tt_err = xt.Table(data=tt_err_data, col_names=tt_raw._col_names)

    # Load rotation table
    tt_raw_rot = xt.Table.from_tfs(fname_rotations)
    tt_rot_data = tt_raw_rot._data.copy()
    tt_rot_data['name'] = np.array([nn.lower() for nn in tt_rot_data['name']])
    tt_rot = xt.Table(data=tt_rot_data, col_names=tt_raw_rot._col_names)
    rot={} # We turn it into a dict for easier access later.
    for nn in tt_rot['name']:
        rot[nn] = {'yrot': tt_rot['yrota', nn], 'srot': tt_rot['srota', nn], 'inout': tt_rot['inout', nn]}

    # Isolate magnets with two apertures (end with .b1, .b2, .v1, .v2)
    tt_err_two_aper = tt_err.rows[r'.*\.b1|.*\.b2|.*\.v1|.*\.v2']

    # I want to keep only the arcs (skip cells 1-7)
    for icell in range(1, 8):
        for ip in [1, 2, 3, 4, 5, 6, 7, 8]:
            tt_err_two_aper = tt_err_two_aper.rows.match_not(
                f'.*\\.{icell}r{ip}.*|.*\\.{icell}l{ip}.*'
                f'|.*\\.a{icell}r{ip}.*|.*\\.a{icell}l{ip}.*'
                f'|.*\\.b{icell}r{ip}.*|.*\\.b{icell}l{ip}.*'
            )

    # Handle rotations and use name with beam instead of name with aper
    name_with_aper = tt_err_two_aper['name']
    name_with_beam = []
    yrotfactor = []
    for nn in name_with_aper:
        # Check if in rot table
        nn_no_aper = nn.replace('.v1', '').replace('.v2', '')
        yyff = 1
        if nn_no_aper in rot: # aperture-beam mapping explicitly given in rot table

            # Check if rotated
            yrot = rot[nn_no_aper]['yrot']
            assert yrot in [0, 180], f"Unexpected yrot value {yrot} for magnet {nn_no_aper}"
            if yrot == 180:
                yyff = -1

            inout = rot[nn_no_aper]['inout']
            assert inout in [1, 2], f"Unexpected inout value {inout} for magnet {nn_no_aper}"
            if (inout == 1 and yyff == 1) or (inout == 2 and yyff == -1):
                nn_with_beam = nn.replace('.v1', '.b1').replace('.v2', '.b2')
            elif (inout == 2 and yyff == 1) or (inout == 1 and yyff == -1):
                nn_with_beam = nn.replace('.v1', '.b2').replace('.v2', '.b1')

        else: # use standard LHC geography (v1 is the external beam)
            side_aper = nn[-5:]
            side_beam = SIDE_APER_TO_SIDE_BEAM[side_aper]
            nn_with_beam = nn[:-5] + side_beam

        if nn_with_beam.endswith('.b2'):
            yyff *= -1  # invert yrot factor for B2 magnets

        name_with_beam.append(nn_with_beam)
        yrotfactor.append(yyff)
    name_with_beam = np.array(name_with_beam)

    tt_err_two_aper['name_with_aper'] = name_with_aper
    tt_err_two_aper['name_with_beam'] = name_with_beam
    tt_err_two_aper['name'] = name_with_beam
    tt_err_two_aper['yrotfactor'] = np.array(yrotfactor)

    # Attach reference order (0 if mb, 1 if mq, fail otherwise)
    main_order = []
    main_is_skew = []
    for nn in tt_err_two_aper['name']:
        for prefix, (order, normal_skew) in PREFIX_TO_MAIN_ORDER:
            assert normal_skew in ['normal', 'skew']
            if nn.startswith(prefix):
                main_order.append(order)
                main_is_skew.append(normal_skew == 'skew')
                break
        else:
            raise ValueError(f"Unexpected magnet name: {nn}")
        # if nn.startswith('mb'):
        #     main_order.append(0)
        # elif nn.startswith('mq'):
        #     main_order.append(1)
        # else:
        #     raise ValueError(f"Unexpected magnet name: {nn}")
    assert len(main_order) == len(tt_err_two_aper)
    assert len(main_is_skew) == len(tt_err_two_aper)

    tt_err_two_aper['main_order'] = np.array(main_order)
    tt_err_two_aper['main_is_skew'] = np.array(main_is_skew)

    # From WISE units to knl_rel and ksl_rel
    ref_radius = 0.017

    knl_rel = np.zeros((len(tt_err_two_aper), max_order))
    ksl_rel = np.zeros((len(tt_err_two_aper), max_order))

    main_order = tt_err_two_aper['main_order']
    yrotfactor = tt_err_two_aper['yrotfactor']
    for ii in range(0, max_order):

        aa = tt_err_two_aper[f'a{ii+1}']
        bb = tt_err_two_aper[f'b{ii+1}']

        # From magnet measurement convention to MADX convention
        dknlr_mad = 1e-4 * bb * (-1 * yrotfactor) ** (main_order + ii    )
        dkslr_mad = 1e-4 * aa * (-1 * yrotfactor) ** (main_order + ii + 1)

        # From MADX convention to knl
        kknn_rel = dknlr_mad * ref_radius**(main_order - (ii)) * factorial(ii) / factorial(main_order)
        kkss_rel = dkslr_mad * ref_radius**(main_order - (ii)) * factorial(ii) / factorial(main_order)

        knl_rel[:, ii] = kknn_rel
        ksl_rel[:, ii] = kkss_rel

    tt_err_two_aper['knl_rel'] = knl_rel
    tt_err_two_aper['ksl_rel'] = ksl_rel

    multipole_errors = {}
    for nn in tt_err_two_aper.name:
        knl_rel = tt_err_two_aper['knl_rel', nn]
        ksl_rel = tt_err_two_aper['ksl_rel', nn]
        main_order = tt_err_two_aper['main_order', nn]
        main_is_skew = tt_err_two_aper['main_is_skew', nn]
        multipole_errors[nn] = {'knl_rel': knl_rel, 'ksl_rel': ksl_rel,
                                'main_order': main_order, 'main_is_skew': main_is_skew}

    # Suppress multipoles of order < 2
    for nn in multipole_errors:
        multipole_errors[nn]['knl_rel'][:min_order] = 0
        multipole_errors[nn]['ksl_rel'][:min_order] = 0

    return multipole_errors, tt_err_two_aper

def set_multipole_errors_in_line(line, multipole_errors,
                                 min_order=2, max_order=15,
                                 error_knob_name=None,
                                 append_order_to_knob_name=True):

    env = line.env

    if error_knob_name:
        if append_order_to_knob_name:
            for ii in range(min_order, max_order):
                env[f'{error_knob_name}_k{ii}'] = 1
                env[f'{error_knob_name}_k{ii}s'] = 1
        else:
            env[error_knob_name] = 1

    # Apply errors in the line
    for nn in line.element_names:
        if not hasattr(line[nn], 'knl'):
            continue  # skip non-multipoles

        print(f'Applying errors to {nn}               ', end='\r', flush=True)
        nn_err = nn.split('..')[0]  # remove ..1, ..2, etc.
        if '/' in nn:
            nn_err = nn_err + '/' + nn.split('/')[1]  # keep /lhcb1 or /lhcb2 if present
        if nn_err in multipole_errors:
            line.extend_knl_rel_ksl_rel(order=max_order, element_names=[nn])
            for ii in range(min_order, max_order):
                kknn_rel = multipole_errors[nn_err]['knl_rel'][ii]
                kkss_rel = multipole_errors[nn_err]['ksl_rel'][ii]
                main_order = int(multipole_errors[nn_err]['main_order'])
                main_is_skew = int(multipole_errors[nn_err]['main_is_skew'])

                if error_knob_name:
                    if append_order_to_knob_name:
                        knob_kn_name = f'{error_knob_name}_k{ii}'
                        knob_ks_name = f'{error_knob_name}_k{ii}s'
                    else:
                        knob_kn_name = error_knob_name
                        knob_ks_name = error_knob_name
                    ref_knob_kn = env.ref[knob_kn_name]
                    ref_knob_ks = env.ref[knob_ks_name]
                else:
                    ref_knob_kn = 1
                    ref_knob_ks = 1

                # Using knl_rel and ksl_rel
                line[nn].main_order = main_order
                line[nn].main_is_skew = main_is_skew
                line.ref[nn].knl_rel[ii] = kknn_rel * ref_knob_kn
                line.ref[nn].ksl_rel[ii] = kkss_rel * ref_knob_ks

