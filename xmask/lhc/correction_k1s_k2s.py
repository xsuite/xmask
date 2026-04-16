import numpy as np
import xtrack as xt

def _correct_k1s_or_k2s(env, twiss_b1, twiss_b2, correct, feed_down=True):

    tw_b1 = twiss_b1
    tw_b2 = twiss_b2

    assert correct in ['k1s', 'k2s']

    arc_names = ['12', '23', '34', '45', '56', '67', '78', '81']

    if correct == 'k2s':
        generated_knob_prefix = 'on_corr_k2s'
        def chrom_coupling_integrand(tw, tt):
            return tt['k2sl'] * tw.dx * np.sqrt(tw.betx * tw.bety) * np.exp(1j*2*np.pi*(tw.mux - tw.muy))
        target_integrand = chrom_coupling_integrand
        correction_knobs = {
            'b1': {arc: [f'kss.a{arc}b1'] for arc in arc_names},
            'b2': {arc: [f'kss.a{arc}b2'] for arc in arc_names},
        }
    elif correct == 'k1s':
        generated_knob_prefix = 'on_corr_k1s'
        target_integrand = 'f1001'
        correction_knobs = {
            'b1': {'12': ['kqs.r1b1', 'kqs.l2b1'], '23': ['kqs.a23b1'],
                '34': ['kqs.r3b1', 'kqs.l4b1'], '45': ['kqs.a45b1'],
                '56': ['kqs.r5b1', 'kqs.l6b1'], '67': ['kqs.a67b1'],
                '78': ['kqs.r7b1', 'kqs.l8b1'], '81': ['kqs.a81b1']},
            'b2': {'12': ['kqs.a12b2'], '23': ['kqs.r2b2', 'kqs.l3b2'],
                '34': ['kqs.a34b2'], '45': ['kqs.r4b2', 'kqs.l5b2'],
                '56': ['kqs.a56b2'], '67': ['kqs.r6b2', 'kqs.l7b2'],
                '78': ['kqs.a78b2'], '81': ['kqs.r8b2', 'kqs.l1b2']},
        }

    tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

    integ_dct = {}

    for beam_name in ['b1', 'b2']:

        if tw_b12[beam_name] is None:
            continue

        line_name = beam_name if beam_name in env.lines else f'lhc{beam_name}'
        line = env[line_name]

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
            for kknn in correction_knobs[beam_name][arc]:
                if kknn not in line.vars:
                    # Some circuits are condemned
                    print(f"Warning: {kknn} not found in line {line_name}, skipping it for the global correction")
                else:
                    correction_knobs_global.append(kknn)
        generated_knob_name = f'{generated_knob_prefix}_global'

        # Create calculator for global correction (not run)
        rdt_contrib_glob = xt.IntegralOptimization(
                                line=line,
                                twiss=tw,
                                feed_down=feed_down,
                                start=start,
                                end=end,
                                vary=xt.VaryList(correction_knobs_global, step=1e-5),
                                target_quantities=target_quantities,
                                generated_knob_name=generated_knob_name)

        # Local correction arc by arc
        opt_dct = {}
        for arc_name in arc_names:
            if beam_name == 'b1':
                start = f's.ds.r{arc_name[0]}.b1'
                end = f'e.ds.l{arc_name[1]}.b1'
            else:
                start = f'e.ds.l{arc_name[1]}.b2'
                end = f's.ds.r{arc_name[0]}.b2'
            # correction_knobs_local = correction_knobs[beam_name][arc_name]
            correction_knobs_local = []
            for kknn in correction_knobs[beam_name][arc_name]:
                if kknn not in line.vars:
                    print(f"Warning: {kknn} not found in line {line_name}, "
                          f"skipping it for the local correction of arc {arc_name}")
                else:
                    correction_knobs_local.append(kknn)

            if len(correction_knobs_local) == 0:
                print(f"None of correction knobs {correction_knobs[beam_name][arc_name]} found "
                      f"for beam {beam_name} arc {arc_name}, skipping local correction for this arc.")
                continue
            generated_knob_name = f'{generated_knob_prefix}_a{arc_name}_local'

            # Usage:
            arc_integ = xt.IntegralOptimization(
                                    line=line,
                                    twiss=tw,
                                    feed_down=feed_down,
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
            integ_dct[f'{beam_name}_{arc_name}'] = arc_integ

        # Global correction
        opt = rdt_contrib_glob.correct()

        integ_dct[f'{beam_name}_global'] = rdt_contrib_glob

        print("Before setting the knob:")
        line.vars.get_table().rows[correction_knobs_global].show()

        env[opt.knob_name] = 1.0
        print("After setting the knob:")
        line.vars.get_table().rows[correction_knobs_global].show()

    return integ_dct

def correct_k1s(env, twiss_b1, twiss_b2, feed_down=True):
    return _correct_k1s_or_k2s(env, twiss_b1, twiss_b2, correct='k1s',
                               feed_down=feed_down)

def correct_k2s(env, twiss_b1, twiss_b2, feed_down=True):
    return _correct_k1s_or_k2s(env, twiss_b1, twiss_b2, correct='k2s',
                               feed_down=feed_down)
