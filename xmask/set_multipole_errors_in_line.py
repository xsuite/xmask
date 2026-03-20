
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

        if '..' in nn: # it's a slice
            nn_err = nn.split('..')[0]  # remove ..1, ..2, etc.
            if '/' in nn:
                nn_err = nn_err + '/' + nn.split('/')[1]  # keep /lhcb1, /lhcb2, /b1, /b2 if present
        else:
            nn_err = nn
        if nn_err in multipole_errors:
            print(f'Applying errors to {nn}               ', end='\r', flush=True)
            line.extend_knl_rel_ksl_rel(order=max_order, element_names=[nn])
            for ii in range(min_order, max_order):
                kknn_rel = (multipole_errors[nn_err]['knl_rel'][ii]
                    if ii < len(multipole_errors[nn_err]['knl_rel']) else 0)
                kkss_rel = (multipole_errors[nn_err]['ksl_rel'][ii]
                    if ii < len(multipole_errors[nn_err]['ksl_rel']) else 0)
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