import numpy as np
import xtrack as xt
from integral_optimization import IntegralOptimization

from corrector_limits import corrector_limits

env = xt.load('lhc_arc_errors.json')

# No dipolar and quadrupolar errors
env['on_error_arc_k0'] = 0
env['on_error_arc_k0s'] = 0
env['on_error_arc_k1'] = 0
env['on_error_arc_k1s'] = 0

# Status of error knobs
tt_err_knobs = env.vars.get_table().rows[r'on_error_arc.*']
print("Error knobs in the environment:")
tt_err_knobs.show()

# Errors off to get reference twiss
env.set(tt_err_knobs.name, 0)
tw_b1 = env['lhcb1'].twiss4d(reverse=False) # Reference twiss
tw_b2 = env['lhcb2'].twiss4d(reverse=False) # Reference twiss
tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

# errors back on
for nn in tt_err_knobs.name:
    env[nn] = tt_err_knobs['value', nn]

arc_multipoles_to_suppress = {
    'k2l': 'kcs',
    'k3l': 'kco',
    'k4l': 'kcd',
}

arcs = ['12', '23', '34', '45', '56', '67', '78', '81']
beams = ['b1', 'b2']

for beam_name in beams:

    line = env[f'lhc{beam_name}']
    tw = tw_b12[beam_name]

    tt = line.get_table(attr=True)
    scale_multipole = np.zeros_like(tt.s)
    scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
    scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors

    for arc_name in arcs:
        # identify range for the arc
        if beam_name == 'b1':
            start = f's.ds.r{arc_name[0]}.{beam_name}'
            end = f'e.ds.l{arc_name[1]}.{beam_name}'
        else:
            start = f'e.ds.l{arc_name[1]}.{beam_name}'
            end = f's.ds.r{arc_name[0]}.{beam_name}'

        for multipole, knob_prefix in arc_multipoles_to_suppress.items():
            correction_knobs = [f'{knob_prefix}.a{arc_name}{beam_name}']
            target_quantities={'multipole_to_suppress': lambda tw, tt: tt[multipole]}

            vary = []
            for kk in correction_knobs:
                vary.append(xt.Vary(kk, step=1e-5, limits=corrector_limits[kk]))

            # Usage:
            rdt_contrib = IntegralOptimization(
                                    line=line,
                                    tw=tw,
                                    start=start,
                                    end=end,
                                    vary=vary,
                                    target_quantities=target_quantities,
                                    generated_knob_name=f'on_corr_{knob_prefix}_arc{arc_name}_{beam_name}',
                                    scale_multipoles=scale_multipole)

            opt = rdt_contrib.correct()

            print("Before setting the knob:")
            line.vars.get_table().rows[correction_knobs].show()

            env[opt.knob_name] = 1.0
            print("After setting the knob:")
            line.vars.get_table().rows[correction_knobs].show()

env.to_json('lhc_arc_errors_with_spool_piece_corrections.json')
