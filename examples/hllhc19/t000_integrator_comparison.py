import xtrack as xt
import numpy as np

lhc = xt.load('./collider_01_multipolar_errors_corrected.json')

lhc.set(lhc.vars.get_table().rows['on_.*'], 0) # Flat machine no errors
lhc.set(lhc.vars.get_table().rows['on_error_.*'], 1) # Switch on all errors
lhc.set(lhc.vars.get_table().rows['on_error_.*_k0'], 0) # k0 errors off
lhc.set(lhc.vars.get_table().rows['on_error_.*_k0s'], 0) # k0s errors off
lhc.set(lhc.vars.get_table().rows['on_error_.*_k1'], 0) # k1 errors off
lhc.set(lhc.vars.get_table().rows['on_error_.*_k1s'], 0) # k1s errors off

lhc.b1.configure_bend_model(core='rot-kick-rot', num_multipole_kicks=1)
print("rot-kick-rot  |x|: ", np.max(np.abs(lhc.b1.twiss4d().x)))

lhc.b1.configure_bend_model(core='rot-kick-rot-simplified', num_multipole_kicks=14)
print("rot-kick-rot-simplified  |x|: ", np.max(np.abs(lhc.b1.twiss4d().x)))