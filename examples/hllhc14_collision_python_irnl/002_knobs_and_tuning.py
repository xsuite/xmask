import xtrack as xt
import xmask as xm

# Load collider anf build trackers
collider = xt.Multiline.from_json('collider_01_bb_off.json')
collider.build_trackers()

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)
conf_knobs_and_tuning = config['config_knobs_and_tuning']

# Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
# experimental magnets, etc.)
for kk, vv in conf_knobs_and_tuning['knob_settings'].items():
    collider.vars[kk] = vv

# Correct Magnetic Errors ---
conf_correct_magnetic_errors = conf_knobs_and_tuning["correct_magnetic_errors"]
if conf_correct_magnetic_errors["enable"]:
    # Pre-tuning
    if conf_correct_magnetic_errors["pre_tuning"]:
        for line_name in ['lhcb1', 'lhcb2']:
            print(f"Performing pre-tuning for {line_name}")
            xm.machine_tuning(line=collider[line_name],
                enable_closed_orbit_correction=True,
                enable_linear_coupling_correction=False,  # !
                enable_tune_correction=True,
                enable_chromaticity_correction=True,
                dual_pass_tune_and_chroma=True,
                dual_pass_tol_factor = 5.,  # add some leeway in the first pass
                knob_names=conf_knobs_and_tuning['knob_names'][line_name],
                targets=conf_knobs_and_tuning["targets"][line_name],
                line_co_ref=collider[f'{line_name}_co_ref'],
                co_corr_config=conf_knobs_and_tuning['closed_orbit_correction'][line_name],
            )

    # Error Correction
    conf_ir_rdt_correction = conf_correct_magnetic_errors.get('ir_rdt_correction', {'enable': False})
    xm.correct_errors(collider,
        enable_ir_rdt_correction=conf_ir_rdt_correction['enable'],
        ir_rdt_corr_config=conf_ir_rdt_correction,
    )

# Full tuning for both lines ---
for line_name in ['lhcb1', 'lhcb2']:
    xm.machine_tuning(line=collider[line_name],
        enable_closed_orbit_correction=True,
        enable_linear_coupling_correction=True,
        enable_tune_correction=True,
        enable_chromaticity_correction=True,
        coupling_correction_analytical_estimation=False,
        coupling_correction_iterative_estimation=0,
        knob_names=conf_knobs_and_tuning['knob_names'][line_name],
        targets=conf_knobs_and_tuning["targets"][line_name],
        line_co_ref=collider[f'{line_name}_co_ref'],
        co_corr_config=conf_knobs_and_tuning['closed_orbit_correction'][line_name],
    )

collider.to_json('collider_02_tuned_bb_off.json')
