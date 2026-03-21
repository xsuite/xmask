import xtrack as xt

def correct_ir_errors(env, twiss_b1, twiss_b2, corrections):

    tw_b1 = twiss_b1
    tw_b2 = twiss_b2

    all_correction_knobs = []
    all_generated_knobs = []
    for ip_name, ip_corr in corrections.items():
        for corr_name, corr in ip_corr['corrections'].items():
            all_correction_knobs += corr['correction_knobs']
            all_generated_knobs.append(corr_name)

    # Clean original values
    for kk in all_correction_knobs:
        env[kk] = 0.0

    for ip_name, ip_corr in corrections.items():
        range_b1 = ip_corr['range_b1']
        range_b2 = ip_corr['range_b2']
        for corr_name, corr in ip_corr['corrections'].items():

            correction_knobs = corr['correction_knobs']
            target_quantities_b1 = corr['target_quantities_b1']
            target_quantities_b2 = corr['target_quantities_b2']
            feed_down = corr['feed_down']
            generated_knob_name = corr_name

            # Usage:
            line_name_b1 = 'b1' if 'b1' in env.lines else 'lhcb1'
            rdt_contrib_b1 = xt.IntegralOptimization(
                                    line=env[line_name_b1],
                                    twiss=tw_b1,
                                    start=range_b1[0],
                                    end=range_b1[1],
                                    vary=xt.VaryList(correction_knobs, step=1e-5),
                                    feed_down=feed_down,
                                    target_quantities=target_quantities_b1,
                                    generated_knob_name=generated_knob_name)

            line_name_b2 = 'b2' if 'b2' in env.lines else 'lhcb2'
            rdt_contrib_b2 = xt.IntegralOptimization(
                                    line=env[line_name_b2],
                                    twiss=tw_b2,
                                    start=range_b2[0],
                                    end=range_b2[1],
                                    vary=[], # only targets here
                                    feed_down=feed_down,
                                    target_quantities=target_quantities_b2,
                                    generated_knob_name=generated_knob_name)

            knob_opt_b1 = rdt_contrib_b1.get_optimizer()
            knob_opt_b2 = rdt_contrib_b2.get_optimizer()

            combined_opt = knob_opt_b1.opt.clone(add_targets=knob_opt_b2.opt.targets)
            combined_opt.step()
            knob_opt_b1.generate_knob()

            env[generated_knob_name] = 1.0
            env.vars.get_table().rows[correction_knobs].show()
