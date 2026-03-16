import numpy as np
import xtrack as xt
from integral_optimization import IntegralOptimization

env = xt.load('lhc_arc_errors_with_spool_piece_corrections.json')

arc_names = ['12', '23', '34', '45', '56', '67', '78', '81']

generated_knob_prefix = 'on_corr_k2sl'
def chrom_coupling_integrand(tw, tt):
    return tt['k2sl'] * tw.dx * np.sqrt(tw.betx * tw.bety) * np.exp(1j*2*np.pi*(tw.mux - tw.muy))
target_integrand = chrom_coupling_integrand
correction_knobs = {
    'b1': {arc: [f'kss.a{arc}b1'] for arc in arc_names},
    'b2': {arc: [f'kss.a{arc}b2'] for arc in arc_names},
}

# generated_knob_prefix = 'on_corr_k1sl'
# target_integrand = 'f1001'
# correction_knobs = {
#     'b1': {'12': ['kqs.r1b1', 'kqs.l2b1'], '23': ['kqs.a23b1'],
#            '34': ['kqs.r3b1', 'kqs.l4b1'], '45': ['kqs.a45b1'],
#            '56': ['kqs.r5b1', 'kqs.l6b1'], '67': ['kqs.a67b1'],
#            '78': ['kqs.r7b1', 'kqs.l8b1'], '81': ['kqs.a81b1']},
#     'b2': {'12': ['kqs.a12b2'], '23': ['kqs.r2b2', 'kqs.l3b2'],
#            '34': ['kqs.a34b2'], '45': ['kqs.r4b2', 'kqs.l5b2'],
#            '56': ['kqs.a56b2'], '67': ['kqs.r6b2', 'kqs.l7b2'],
#            '78': ['kqs.a78b2'], '81': ['kqs.r8b2', 'kqs.l1b2']},
# }
# env['on_error_arc_k1s'] = 1.0 # I enable the skew quad errors (which are off by default in the input file)

# Status of error and correction knobs
tt_err_knobs = env.vars.get_table().rows[r'on_error_arc.*|on_corr_.*']

# Errors and corrections off to get reference twiss
env.set(tt_err_knobs.name, 0)
tw_b1 = env['lhcb1'].twiss4d(reverse=False) # Reference twiss
tw_b2 = env['lhcb2'].twiss4d(reverse=False) # Reference twiss
tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

# errors and corrections back on
for nn in tt_err_knobs.name:
    env[nn] = tt_err_knobs['value', nn]

for beam_name in ['b1', 'b2']:

    line = env[f'lhc{beam_name}']

    tw = tw_b12[beam_name]

    # Function that we want to minimize
    if isinstance(target_integrand, str):
        target_quantities = {'target': target_integrand}
    else:
        # Assume a complex callable
        target_quantities={
            'target_real': lambda tw, tt: np.real(target_integrand(tw, tt)),
            'target_imag': lambda tw, tt: np.imag(target_integrand(tw, tt))
        }


    # Global correction setup (we run it after local)
    start = tw.name[0]
    end = tw.name[-1]
    correction_knobs_global = []
    for arc in arc_names:
        correction_knobs_global += correction_knobs[beam_name][arc]
    generated_knob_name = f'{generated_knob_prefix}_global'

    # Create calculator for global correction (not run)
    rdt_contrib_glob = IntegralOptimization(
                            line=line,
                            tw=tw,
                            feed_down=True,
                            start=start,
                            end=end,
                            vary=xt.VaryList(correction_knobs_global, step=1e-5),
                            target_quantities=target_quantities,
                            generated_knob_name=generated_knob_name)

    # Local correction arc by arc
    opt_dct = {}
    integ_dct = {}
    for arc_name in arc_names:
        if beam_name == 'b1':
            start = f's.ds.r{arc_name[0]}.b1'
            end = f'e.ds.l{arc_name[1]}.b1'
        else:
            start = f'e.ds.l{arc_name[1]}.b2'
            end = f's.ds.r{arc_name[0]}.b2'
        correction_knobs_local = correction_knobs[beam_name][arc_name]
        generated_knob_name = f'{generated_knob_prefix}_a{arc_name}_local'

        # Usage:
        arc_integ = IntegralOptimization(
                                line=line,
                                tw=tw,
                                feed_down=True,
                                start=start,
                                end=end,
                                vary=xt.VaryList(correction_knobs_local, step=1e-5),
                                target_quantities=target_quantities,
                                generated_knob_name=generated_knob_name)

        opt = arc_integ.correct()
        print("Before setting the knob:")
        line.vars.get_table().rows[correction_knobs_local].show()

        env[opt.knob_name] = 1.0
        print("After setting the knob:")
        line.vars.get_table().rows[correction_knobs_local].show()

        opt_dct[arc_name] = opt
        integ_dct[arc_name] = arc_integ

    # Global correction
    opt = rdt_contrib_glob.correct()

    print("Before setting the knob:")
    line.vars.get_table().rows[correction_knobs_global].show()

    env[opt.knob_name] = 1.0
    print("After setting the knob:")
    line.vars.get_table().rows[correction_knobs_global].show()

env.to_json('lhc_arc_errors_with_correction.json')
