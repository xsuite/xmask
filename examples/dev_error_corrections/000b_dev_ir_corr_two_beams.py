import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('collider_00_from_mad_with_errors.json')

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw_b1 = env_no_err.lhcb1.twiss4d(reverse=False) # Reference twiss
tw_b2 = env_no_err.lhcb2.twiss4d(reverse=False) # Reference twiss

tt0_b1 = env.lhcb1.get_table(attr=True)
tt0_b2 = env.lhcb2.get_table(attr=True)


# # Normal octupole correction ip5
# generated_knob_name='on_corr_k3_ip5'
# correction_knobs=['kcox3.l5', 'kcox3.r5']
# multipole='k3l'
# target_quantities_b1={'f4000_b1': 'f4000'}
# target_quantities_b2={'f4000_b2': 'f4000'}
# range_b1 = 'dfxj.4l5', 'dfxj.4r5'
# range_b2 = 'dfxj.4r5', 'dfxj.4l5'
# feed_down = False # to have same result as legacy

# # Normal sextupole correction ip5
# generated_knob_name='on_corr_k2_ip5'
# correction_knobs=['kcsx3.l5', 'kcsx3.r5']
# multipole='k2l'
# target_quantities_b1={'f1020_b1': 'f1020'}
# target_quantities_b2={'f1020_b2': 'f1020'}
# range_b1 = 'dfxj.4l5', 'dfxj.4r5'
# range_b2 = 'dfxj.4r5', 'dfxj.4l5'
# feed_down = False # to have same result as legacy

ir_corrections = {
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
        }
    }
}

ip_name = 'ip5'
correction_name = 'on_corr_k5_ip5'

ip_corrections = ir_corrections[ip_name]
range_b1 = ip_corrections['range_b1']
range_b2 = ip_corrections['range_b2']
corrections = ip_corrections['corrections']
correction = corrections[correction_name]
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
                         ip=None, # not needed when RDT are used
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
                         ip=None, # not needed when RDT are used
                         feed_down=feed_down,
                         target_quantities=target_quantities_b2,
                         generated_knob_name=generated_knob_name)

print("Original correction:")
rdt_contrib_b1.print_corrections()
rdt_contrib_b1.clear_corrections()

knob_opt_b1 = rdt_contrib_b1.get_optimizer()
knob_opt_b2 = rdt_contrib_b2.get_optimizer()

# opt = rdt_contrib_b1.correct() # correct only b1
combined_opt = knob_opt_b1.opt.clone(add_targets=knob_opt_b2.opt.targets)
combined_opt.step()
knob_opt_b1.generate_knob()

print("Before setting the knob:")
rdt_contrib_b1.print_corrections()

env[knob_opt_b1.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib_b1.print_corrections()

