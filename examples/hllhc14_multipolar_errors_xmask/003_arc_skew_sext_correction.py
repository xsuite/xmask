
import numpy as np
import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('lhc_arc_errors_with_spool_piece_corrections.json')

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

import matplotlib.pyplot as plt
plt.close('all')

for beam_name in ['b1', 'b2']:

    line = env[f'lhc{beam_name}']

    tw = tw_b12[beam_name]

    tt = line.get_table()
    scale_multipole = np.zeros_like(tt.s)
    scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
    scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors
    scale_multipole[tt.rows.mask[r'mss.*']] = 1.0 # all magnets called mssXXX used as correctors

    def chorm_coupling_integrand(tw, tt):
        return tt['k2sl'] * tw.dx * np.sqrt(tw.betx * tw.bety) * np.exp(1j*2*np.pi*(tw.mux - tw.muy))

    arcs = ['12', '23', '34', '45', '56', '67', '78', '81']

    # Global correction setup (we run it after logal)
    start = tw.name[0]
    end = tw.name[-1]
    correction_knobs = [f'kss.a{arc_name}{beam_name}' for arc_name in arcs]
    generated_knob_name = f'on_corr_k2sl_global'
    multipole = 'k2sl'
    target_quantities={
        'chrom_coupling_real': lambda tw, tt: chorm_coupling_integrand(tw, tt).real,
        'chrom_coupling_imag': lambda tw, tt: chorm_coupling_integrand(tw, tt).imag
    }

    rdt_contrib_glob = IntegralCorrection(
                            line=line,
                            tw=tw,
                            feed_down=True,
                            start=start,
                            end=end,
                            correction_knobs=correction_knobs,
                            multipole=multipole,
                            target_quantities=target_quantities,
                            generated_knob_name=generated_knob_name,
                            scale_multipole=scale_multipole)

    opt_dct = {}
    integ_dct = {}
    for arc_name in arcs:
        if beam_name == 'b1':
            start = f's.ds.r{arc_name[0]}.b1'
            end = f'e.ds.l{arc_name[1]}.b1'
        else:
            start = f'e.ds.l{arc_name[1]}.b2'
            end = f's.ds.r{arc_name[0]}.b2'
        correction_knobs = [f'kss.a{arc_name}{beam_name}']
        generated_knob_name = f'on_corr_k2sl_a{arc_name}_local'
        multipole = 'k2sl'
        target_quantities={
            'chrom_coupling_real': lambda tw, tt: chorm_coupling_integrand(tw, tt).real,
            'chrom_coupling_imag': lambda tw, tt: chorm_coupling_integrand(tw, tt).imag
        }

        # Usage:
        arc_integ = IntegralCorrection(
                                line=line,
                                tw=tw,
                                feed_down=True,
                                start=start,
                                end=end,
                                correction_knobs=correction_knobs,
                                multipole=multipole,
                                target_quantities=target_quantities,
                                generated_knob_name=generated_knob_name,
                                scale_multipole=scale_multipole)

        opt = arc_integ.correct()
        print("Before setting the knob:")
        arc_integ.print_corrections()

        env[opt.knob_name] = 1.0
        print("After setting the knob:")
        arc_integ.print_corrections()

        opt_dct[arc_name] = opt
        integ_dct[arc_name] = arc_integ

    # Global correction

    opt = rdt_contrib_glob.correct()

    print("Before setting the knob:")
    rdt_contrib_glob.print_corrections()

    env[opt.knob_name] = 1.0
    print("After setting the knob:")
    knobs_final = rdt_contrib_glob.get_corrections()
    rdt_contrib_glob.print_corrections()

    local_dct_opt = {}
    for arc_name in arcs:
        oo = integ_dct[arc_name].run()
        local_dct_opt[arc_name] = oo['chrom_coupling_real'] + 1j * oo['chrom_coupling_imag']

    tw_opt = line.twiss4d(coupling_edw_teng=True, delta0=1e-3)
    nlchr_opt = line.get_non_linear_chromaticity(num_delta=20)

    # Compare in a bar prot the abs of the local integrals before and after the global correction

    plt.figure(1+int(beam_name[1:])*10)
    arc_names = list(opt_dct.keys())
    local_opt_vals = [np.abs(local_dct_opt[arc_name]) for arc_name in arc_names]
    x = np.arange(len(arc_names))
    plt.bar(x, local_opt_vals, width=0.2, label='After global correction')
    plt.legend()

    plt.xticks(x, arc_names)
    plt.xlabel('Arc name')
    plt.ylabel('Abs of chromatic coupling integral')

    plt.figure(2+int(beam_name[1:])*10)
    plt.plot(nlchr_opt.delta0, [ttt.c_minus for ttt in nlchr_opt.twiss], label='After global correction')
    plt.xlabel('delta0')
    plt.ylabel('C-')
    plt.legend()

env.to_json('lhc_arc_errors_with_correction.json')

plt.show()