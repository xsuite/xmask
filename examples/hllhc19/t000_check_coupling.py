import xtrack as xt

lhc = xt.load("lhc_thick_test_04_tuned_and_leveled_bb_on.json")

tt_vars = lhc.vars.get_table(expr_obj=True)

print("Final config - |C-| = ", lhc['b1'].twiss4d().c_minus)
lhc['beambeam_scale'] = 0
print("Beam beam off - |C-| = ", lhc['b1'].twiss4d().c_minus)

lhc.set(tt_vars.rows['on_.*'], 0)
lhc.set(tt_vars.rows['cm.*'], 0)
print("Bumps, errors and corrections off - |C-| = ", lhc['b1'].twiss4d().c_minus)

# Errors back on
tt_err_restore = tt_vars.rows['on_error_.*']
for nn in tt_err_restore.name:
    lhc[nn] = tt_err_restore['value', nn]
print("Errors back on (flat machine) - |C-| = ", lhc['b1'].twiss4d().c_minus)

# Corrections back on
tt_corr_restore = tt_vars.rows[r'on_corr_k.*']
for nn in tt_corr_restore.name:
    lhc[nn] = tt_corr_restore['value', nn]
print("Corrections back on (flat machine) - |C-| = ", lhc['b1'].twiss4d().c_minus)