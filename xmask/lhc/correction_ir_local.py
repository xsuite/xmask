import xtrack as xt

DEFAULT_IR15_CORRECTIONS = {
    'ip1': {'range_b1': ['dfxj.4l1', 'dfxj.4r1'],'range_b2': ['dfxj.4r1', 'dfxj.4l1'],
            'corrections': {
                'on_corr_k2_ip1': {
                    'correction_knobs': ['kcsx3.l1', 'kcsx3.r1'],
                    'target_quantities_b1': {'f1020_b1': 'f1020'},
                    'target_quantities_b2': {'f1020_b2': 'f1020'},
                    'feed_down': False},
                'on_corr_k3_ip1': {
                    'correction_knobs': ['kcox3.l1', 'kcox3.r1'],
                    'target_quantities_b1': {'f4000_b1': 'f4000'},
                    'target_quantities_b2': {'f4000_b2': 'f4000'},
                    'feed_down': False},
                'on_corr_k4_ip1': {
                    'correction_knobs': ['kcdx3.l1', 'kcdx3.r1'],
                    'target_quantities_b1': {'f5000_b1': 'f5000'},
                    'target_quantities_b2': {'f5000_b2': 'f5000'},
                    'feed_down': False},
                'on_corr_k5_ip1': {
                    'correction_knobs': ['kctx3.l1', 'kctx3.r1'],
                    'target_quantities_b1': {'f6000_b1': 'f6000'},
                    'target_quantities_b2': {'f6000_b2': 'f6000'},
                    'feed_down': False},
                'on_corr_k1s_ip1': {
                    'correction_knobs': ['kqsx3.l1', 'kqsx3.r1'],
                    'target_quantities_b1': {'f1001_b1': 'f1001'},
                    'target_quantities_b2': {'f1001_b2': 'f1001'},
                    'feed_down': False},
                'on_corr_k2s_ip1': {
                    'correction_knobs': ['kcssx3.l1', 'kcssx3.r1'],
                    'target_quantities_b1': {'f0030_b1': 'f0030'},
                    'target_quantities_b2': {'f0030_b2': 'f0030'},
                    'feed_down': False},
                'on_corr_k3s_ip1': {
                    'correction_knobs': ['kcosx3.l1', 'kcosx3.r1'],
                    'target_quantities_b1': {'f1030_b1': 'f1030'},
                    'target_quantities_b2': {'f1030_b2': 'f1030'},
                    'feed_down': False},
                'on_corr_k4s_ip1': {
                    'correction_knobs': ['kcdsx3.l1', 'kcdsx3.r1'],
                    'target_quantities_b1': {'f0050_b1': 'f0050'},
                    'target_quantities_b2': {'f0050_b2': 'f0050'},
                    'feed_down': False},
                'on_corr_k5s_ip1': {
                    'correction_knobs': ['kctsx3.l1', 'kctsx3.r1'],
                    'target_quantities_b1': {'f1050_b1': 'f1050'},
                    'target_quantities_b2': {'f1050_b2': 'f1050'},
                    'feed_down': False}
        }
    },
    'ip5': {'range_b1': ['dfxj.4l5', 'dfxj.4r5'],'range_b2': ['dfxj.4r5', 'dfxj.4l5'],
            'corrections': {
                'on_corr_k2_ip5': {
                    'correction_knobs': ['kcsx3.l5', 'kcsx3.r5'],
                    'target_quantities_b1': {'f1020_b1': 'f1020'},
                    'target_quantities_b2': {'f1020_b2': 'f1020'},
                    'feed_down': False},
                'on_corr_k3_ip5': {
                    'correction_knobs': ['kcox3.l5', 'kcox3.r5'],
                    'target_quantities_b1': {'f4000_b1': 'f4000'},
                    'target_quantities_b2': {'f4000_b2': 'f4000'},
                    'feed_down': False},
                'on_corr_k4_ip5': {
                    'correction_knobs': ['kcdx3.l5', 'kcdx3.r5'],
                    'target_quantities_b1': {'f5000_b1': 'f5000'},
                    'target_quantities_b2': {'f5000_b2': 'f5000'},
                    'feed_down': False},
                'on_corr_k5_ip5': {
                    'correction_knobs': ['kctx3.l5', 'kctx3.r5'],
                    'target_quantities_b1': {'f6000_b1': 'f6000'},
                    'target_quantities_b2': {'f6000_b2': 'f6000'},
                    'feed_down': False},
                'on_corr_k1s_ip5': {
                    'correction_knobs': ['kqsx3.l5', 'kqsx3.r5'],
                    'target_quantities_b1': {'f1001_b1': 'f1001'},
                    'target_quantities_b2': {'f1001_b2': 'f1001'},
                    'feed_down': False},
                'on_corr_k2s_ip5': {
                    'correction_knobs': ['kcssx3.l5', 'kcssx3.r5'],
                    'target_quantities_b1': {'f0030_b1': 'f0030'},
                    'target_quantities_b2': {'f0030_b2': 'f0030'},
                    'feed_down': False},
                'on_corr_k3s_ip5': {
                    'correction_knobs': ['kcosx3.l5', 'kcosx3.r5'],
                    'target_quantities_b1': {'f1030_b1': 'f1030'},
                    'target_quantities_b2': {'f1030_b2': 'f1030'},
                    'feed_down': False},
                'on_corr_k4s_ip5': {
                    'correction_knobs': ['kcdsx3.l5', 'kcdsx3.r5'],
                    'target_quantities_b1': {'f0050_b1': 'f0050'},
                    'target_quantities_b2': {'f0050_b2': 'f0050'},
                    'feed_down': False},
                'on_corr_k5s_ip5': {
                    'correction_knobs': ['kctsx3.l5', 'kctsx3.r5'],
                    'target_quantities_b1': {'f1050_b1': 'f1050'},
                    'target_quantities_b2': {'f1050_b2': 'f1050'},
                    'feed_down': False}
        }
    },
}

def correct_ir_errors(env, twiss_b1, twiss_b2, corrections):

    tw_b1 = twiss_b1
    tw_b2 = twiss_b2

    all_correction_knobs = []
    all_generated_knobs = []
    for ip_name, ip_corr in corrections.items():
        for corr_name, corr in ip_corr['corrections'].items():
            all_correction_knobs += corr['correction_knobs']
            all_generated_knobs.append(corr_name)

    original_values = {kk: env[kk] for kk in all_correction_knobs}

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
            rdt_contrib_b1 = xt.IntegralOptimization(
                                    line=env['lhcb1'],
                                    twiss=tw_b1,
                                    start=range_b1[0],
                                    end=range_b1[1],
                                    vary=xt.VaryList(correction_knobs, step=1e-5),
                                    feed_down=feed_down,
                                    target_quantities=target_quantities_b1,
                                    generated_knob_name=generated_knob_name)

            rdt_contrib_b2 = xt.IntegralOptimization(
                                    line=env['lhcb2'],
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

    # switch on all generated knobs
    for kk in all_generated_knobs:
        env[kk] = 1.0
