import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('../hllhc14_multipolar_errors_legacy/collider_errors_on_corrections_on.json')

# Reference twiss
env_no_err = xt.load('../hllhc14_multipolar_errors_legacy/collider_errors_off_corrections_off.json')
tw_b1 = env_no_err.lhcb1.twiss4d(reverse=False) # Reference twiss
tw_b2 = env_no_err.lhcb2.twiss4d(reverse=False) # Reference twiss

tt0_b1 = env.lhcb1.get_table(attr=True)
tt0_b2 = env.lhcb2.get_table(attr=True)


ir_corrections = {
    'ip1': {'range_b1': ['dfxj.4l1', 'dfxj.4r1'],'range_b2': ['dfxj.4r1', 'dfxj.4l1'],
            'corrections': {
                'on_corr_k2_ip1': {
                    'correction_knobs': ['kcsx3.l1', 'kcsx3.r1'], 'multipole': 'k2l',
                    'target_quantities_b1': {'f1020_b1': 'f1020'},
                    'target_quantities_b2': {'f1020_b2': 'f1020'},
                    'feed_down': False},
                'on_corr_k3_ip1': {
                    'correction_knobs': ['kcox3.l1', 'kcox3.r1'], 'multipole': 'k3l',
                    'target_quantities_b1': {'f4000_b1': 'f4000'},
                    'target_quantities_b2': {'f4000_b2': 'f4000'},
                    'feed_down': False},
                'on_corr_k4_ip1': {
                    'correction_knobs': ['kcdx3.l1', 'kcdx3.r1'], 'multipole': 'k4l',
                    'target_quantities_b1': {'f5000_b1': 'f5000'},
                    'target_quantities_b2': {'f5000_b2': 'f5000'},
                    'feed_down': False},
                'on_corr_k5_ip1': {
                    'correction_knobs': ['kctx3.l1', 'kctx3.r1'], 'multipole': 'k5l',
                    'target_quantities_b1': {'f6000_b1': 'f6000'},
                    'target_quantities_b2': {'f6000_b2': 'f6000'},
                    'feed_down': False},
                'on_corr_k1s_ip1': {
                    'correction_knobs': ['kqsx3.l1', 'kqsx3.r1'], 'multipole': 'k1sl',
                    'target_quantities_b1': {'f1001_b1': 'f1001'},
                    'target_quantities_b2': {'f1001_b2': 'f1001'},
                    'feed_down': False},
                'on_corr_k2s_ip1': {
                    'correction_knobs': ['kcssx3.l1', 'kcssx3.r1'], 'multipole': 'k2sl',
                    'target_quantities_b1': {'f0030_b1': 'f0030'},
                    'target_quantities_b2': {'f0030_b2': 'f0030'},
                    'feed_down': False},
                'on_corr_k3s_ip1': {
                    'correction_knobs': ['kcosx3.l1', 'kcosx3.r1'], 'multipole': 'k3sl',
                    'target_quantities_b1': {'f1030_b1': 'f1030'},
                    'target_quantities_b2': {'f1030_b2': 'f1030'},
                    'feed_down': False},
                'on_corr_k4s_ip1': {
                    'correction_knobs': ['kcdsx3.l1', 'kcdsx3.r1'], 'multipole': 'k4sl',
                    'target_quantities_b1': {'f0050_b1': 'f0050'},
                    'target_quantities_b2': {'f0050_b2': 'f0050'},
                    'feed_down': False},
                'on_corr_k5s_ip1': {
                    'correction_knobs': ['kctsx3.l1', 'kctsx3.r1'], 'multipole': 'k5sl',
                    'target_quantities_b1': {'f1050_b1': 'f1050'},
                    'target_quantities_b2': {'f1050_b2': 'f1050'},
                    'feed_down': False}
        }
    },
    'ip5': {'range_b1': ['dfxj.4l5', 'dfxj.4r5'],'range_b2': ['dfxj.4r5', 'dfxj.4l5'],
            'corrections': {
                'on_corr_k2_ip5': {
                    'correction_knobs': ['kcsx3.l5', 'kcsx3.r5'], 'multipole': 'k2l',
                    'target_quantities_b1': {'f1020_b1': 'f1020'},
                    'target_quantities_b2': {'f1020_b2': 'f1020'},
                    'feed_down': False},
                'on_corr_k3_ip5': {
                    'correction_knobs': ['kcox3.l5', 'kcox3.r5'], 'multipole': 'k3l',
                    'target_quantities_b1': {'f4000_b1': 'f4000'},
                    'target_quantities_b2': {'f4000_b2': 'f4000'},
                    'feed_down': False},
                'on_corr_k4_ip5': {
                    'correction_knobs': ['kcdx3.l5', 'kcdx3.r5'], 'multipole': 'k4l',
                    'target_quantities_b1': {'f5000_b1': 'f5000'},
                    'target_quantities_b2': {'f5000_b2': 'f5000'},
                    'feed_down': False},
                'on_corr_k5_ip5': {
                    'correction_knobs': ['kctx3.l5', 'kctx3.r5'], 'multipole': 'k5l',
                    'target_quantities_b1': {'f6000_b1': 'f6000'},
                    'target_quantities_b2': {'f6000_b2': 'f6000'},
                    'feed_down': False},
                'on_corr_k1s_ip5': {
                    'correction_knobs': ['kqsx3.l5', 'kqsx3.r5'], 'multipole': 'k1sl',
                    'target_quantities_b1': {'f1001_b1': 'f1001'},
                    'target_quantities_b2': {'f1001_b2': 'f1001'},
                    'feed_down': False},
                'on_corr_k2s_ip5': {
                    'correction_knobs': ['kcssx3.l5', 'kcssx3.r5'], 'multipole': 'k2sl',
                    'target_quantities_b1': {'f0030_b1': 'f0030'},
                    'target_quantities_b2': {'f0030_b2': 'f0030'},
                    'feed_down': False},
                'on_corr_k3s_ip5': {
                    'correction_knobs': ['kcosx3.l5', 'kcosx3.r5'], 'multipole': 'k3sl',
                    'target_quantities_b1': {'f1030_b1': 'f1030'},
                    'target_quantities_b2': {'f1030_b2': 'f1030'},
                    'feed_down': False},
                'on_corr_k4s_ip5': {
                    'correction_knobs': ['kcdsx3.l5', 'kcdsx3.r5'], 'multipole': 'k4sl',
                    'target_quantities_b1': {'f0050_b1': 'f0050'},
                    'target_quantities_b2': {'f0050_b2': 'f0050'},
                    'feed_down': False},
                'on_corr_k5s_ip5': {
                    'correction_knobs': ['kctsx3.l5', 'kctsx3.r5'], 'multipole': 'k5sl',
                    'target_quantities_b1': {'f1050_b1': 'f1050'},
                    'target_quantities_b2': {'f1050_b2': 'f1050'},
                    'feed_down': False}
        }
    },
    'ip2': {'range_b1': ['bpmsx.4l2.b1', 'bpmsx.4r2.b1'],'range_b2': ['bpmsx.4r2.b2', 'bpmsx.4l2.b2'],
            'corrections': {} # not corrected in the original script
    },
    'ip8': {'range_b1': ['bpmsx.4l8.b1', 'bpmsx.4r8.b1'],'range_b2': ['bpmsx.4r8.b2', 'bpmsx.4l8.b2'],
            'corrections': {} # not corrected in the original script
    }
}

all_correction_knobs = []
all_generated_knobs = []
for ip_name, ip_corrections in ir_corrections.items():
    for correction_name, correction in ip_corrections['corrections'].items():
        all_correction_knobs += correction['correction_knobs']
        all_generated_knobs.append(correction_name)

original_values = {kk: env[kk] for kk in all_correction_knobs}

# Clean original values
for kk in all_correction_knobs:
    env[kk] = 0.0

for ip_name, ip_corrections in ir_corrections.items():
    ip_corrections = ir_corrections[ip_name]
    range_b1 = ip_corrections['range_b1']
    range_b2 = ip_corrections['range_b2']
    for correction_name, correction in ip_corrections['corrections'].items():

        correction_knobs = correction['correction_knobs']
        multipole = correction['multipole']
        target_quantities_b1 = correction['target_quantities_b1']
        target_quantities_b2 = correction['target_quantities_b2']
        feed_down = correction['feed_down']
        generated_knob_name = correction_name

        # Usage:
        rdt_contrib_b1 = IntegralCorrection(
                                line=env['lhcb1'],
                                tw=tw_b1,
                                start=range_b1[0],
                                end=range_b1[1],
                                correction_knobs=correction_knobs,
                                multipole=multipole,
                                feed_down=feed_down,
                                target_quantities=target_quantities_b1,
                                generated_knob_name=generated_knob_name)

        rdt_contrib_b2 = IntegralCorrection(
                                line=env['lhcb2'],
                                tw=tw_b2,
                                start=range_b2[0],
                                end=range_b2[1],
                                correction_knobs=[], # only targets here
                                multipole=multipole,
                                feed_down=feed_down,
                                target_quantities=target_quantities_b2,
                                generated_knob_name=generated_knob_name)

        knob_opt_b1 = rdt_contrib_b1.get_optimizer()
        knob_opt_b2 = rdt_contrib_b2.get_optimizer()

        combined_opt = knob_opt_b1.opt.clone(add_targets=knob_opt_b2.opt.targets)
        combined_opt.step()
        knob_opt_b1.generate_knob()

# assert all generated knobs are off and all correction knobs are off
for kk in all_correction_knobs:
    assert env[kk] == 0.0, f"Correction knob {kk} is not zero at the end of the test"

for kk in all_generated_knobs:
    assert env[kk] == 0.0, f"Generated knob {kk} is not zero at the end of the test"

# switch on all generated knobs
for kk in all_generated_knobs:
    env[kk] = 1.0

# check similarity with original values
import xobjects as xo
for nn, vv in original_values.items():
    xo.assert_allclose(env[nn], vv, rtol=8e-2, atol=1e-12)