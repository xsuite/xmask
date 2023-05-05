import numpy as np

from cpymad.madx import Madx
import xtrack as xt

import xmask as xm
import xmask.lhc as xmlhc
import yaml

# Import user-defined optics-specific tools
from _complementary_run3_ions import _config_ion_yaml_str, build_sequence, apply_optics

# Read config file
config = yaml.safe_load(_config_ion_yaml_str)
config_mad_model = config['config_mad']

# Make mad environment
xm.make_mad_environment(links=config_mad_model['links'])

# Start mad
mad_b1b2 = Madx(command_log="mad_collider.log")
mad_b4 = Madx(command_log="mad_b4.log")

# Build sequences
build_sequence(mad_b1b2, mylhcbeam=1)
build_sequence(mad_b4, mylhcbeam=4)

# Apply optics (only for b1b2, b4 will be generated from b1b2)
apply_optics(mad_b1b2, optics_file=config_mad_model['optics_file'])

# Build xsuite collider
collider = xmlhc.build_xsuite_collider(
    sequence_b1=mad_b1b2.sequence.lhcb1,
    sequence_b2=mad_b1b2.sequence.lhcb2,
    sequence_b4=mad_b4.sequence.lhcb2,
    beam_config=config_mad_model['beam_config'],
    enable_imperfections=config_mad_model['enable_imperfections'],
    enable_knob_synthesis=config_mad_model['enable_knob_synthesis'],
    pars_for_imperfections=config_mad_model['pars_for_imperfections'],
    ver_lhc_run=config_mad_model['ver_lhc_run'],
    ver_hllhc_optics=config_mad_model['ver_hllhc_optics'])


assert len(collider.lines.keys()) == 4

for line_name in collider.lines.keys():
    pref = collider[line_name].particle_ref
    assert np.isclose(pref.q0, 82, rtol=1e-10, atol=0)
    assert np.isclose(pref.energy0[0], 5.74e14, rtol=1e-10, atol=0)
    assert np.isclose(pref.mass0, 193687272900.0, rtol=1e-10, atol=0)
    assert np.isclose(pref.gamma0[0], 2963.54, rtol=1e-6, atol=0)

# Save to file
collider.to_json('collider_lhc_ion_00_from_mad.json')




