import xtrack as xt
import xmask.lhc as xmlhc

env = xt.load('../hllhc14_multipolar_errors_legacy/collider_errors_on_corrections_on.json')

# Reference twiss
env_no_err = xt.load('../hllhc14_multipolar_errors_legacy/collider_errors_off_corrections_off.json')
tw_b1 = env_no_err.lhcb1.twiss4d(reverse=False) # Reference twiss
tw_b2 = env_no_err.lhcb2.twiss4d(reverse=False) # Reference twiss

all_correction_knobs = []
all_generated_knobs = []
for ip_name, ip_corrections in xmlhc.DEFAULT_IR15_CORRECTIONS.items():
    for correction_name, correction in ip_corrections['corrections'].items():
        all_correction_knobs += correction['correction_knobs']
        all_generated_knobs.append(correction_name)

original_values = {kk: env[kk] for kk in all_correction_knobs}

# Clean original values
for kk in all_correction_knobs:
    env[kk] = 0.0

# Local correction of IR15 multipole errors
xmlhc.correct_ir_errors(env, twiss_b1=tw_b1, twiss_b2=tw_b2,
                        corrections=xmlhc.DEFAULT_IR15_CORRECTIONS)

# check similarity with original values
import xobjects as xo
for nn, vv in original_values.items():
    xo.assert_allclose(env[nn], vv, rtol=8e-2, atol=1e-12)