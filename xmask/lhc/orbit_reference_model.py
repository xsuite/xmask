import xdeps as xd
import xmask as xm


def build_closed_orbit_reference(lhc):

    lhc_ref = lhc.copy() # deep copy
    lhc_ref._var_management = None
    lhc_ref._init_var_management() # kills all knobs on the elements

    xm.transfer_vars_to_env(lhc, lhc_ref)

    tt_ref = lhc_ref.elements.get_table()
    tt_correctors = tt_ref.rows.match(name='mcb.*')
    tt_experimental_magnets = tt_ref.rows.match(name=r'mb[xlaw].*\.1[rl][28]/.*')
    element_names_transfer_strengths = list(set(tt_correctors.name) | set(tt_experimental_magnets.name))

    old_default_to_zero = lhc_ref.vars.default_to_zero
    lhc_ref.vars.default_to_zero = True

    formatter = xd.refs.CompactFormatter(scope=None)
    for nn in element_names_transfer_strengths:
        if hasattr(lhc_ref[nn], 'knl'):
            expr_knl = lhc.ref[nn].knl[0]._expr
            if expr_knl is not None:
                lhc_ref[nn].knl[0] = expr_knl._formatted(formatter)
        if hasattr(lhc_ref[nn], 'ksl'):
            expr_ksl = lhc.ref[nn].ksl[0]._expr
            if expr_ksl is not None:
                lhc_ref[nn].ksl[0] = expr_ksl._formatted(formatter)
    lhc_ref.vars.default_to_zero = old_default_to_zero

    return lhc_ref