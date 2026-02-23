
import numpy as np
import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('collider_00_from_mad_with_errors.json')

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw_b1 = env_no_err['lhcb1'].twiss4d(reverse=False) # Reference twiss
tw_b2 = env_no_err['lhcb2'].twiss4d(reverse=False) # Reference twiss

tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

# Let's have a look at the b5 (k4) # decapole

arc_multipoles_to_suppress = {
    # 'k2l': 'kcs',
    'k3l': 'kco',
    # 'k4l': 'kcd',
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
            target_quantities={'multipole_to_suppress': lambda tw, tt: tt[multipole].sum()}

            # Usage:
            rdt_contrib = IntegralCorrection(
                                    line=line,
                                    tw=tw,
                                    start=start,
                                    end=end,
                                    correction_knobs=correction_knobs,
                                    multipole=multipole,
                                    ip=None,
                                    target_quantities=target_quantities,
                                    generated_knob_name=f'on_corr_{knob_prefix}_arc{arc_name}_{beam_name}',
                                    scale_multipole=scale_multipole)
            print()
            print("Original correction:")
            rdt_contrib.print_corrections()

            rdt_contrib.clear_corrections()
            opt = rdt_contrib.correct()

            print("Before setting the knob:")
            rdt_contrib.print_corrections()

            env[opt.knob_name] = 1.0
            print("After setting the knob:")
            rdt_contrib.print_corrections()
