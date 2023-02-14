import json
import yaml

from cpymad.madx import Madx

import xobjects as xo

import pymaskmx as pm
import pymaskmx.lhc as pmlhc

# Import user-defined optics-specific tools
import optics_specific_tools_hlhc14 as ost

# Read config file
with open('config_mad.yaml','r') as fid:
    configuration = yaml.safe_load(fid)

# Make mad environment
pm.make_mad_environment(links=configuration['links'])

# Start mad
mad_b1b2 = Madx(command_log="mad_collider.log")
mad_b4 = Madx(command_log="mad_b4.log")

# Build sequences
ost.build_sequence(mad_b1b2, mylhcbeam=1)
ost.build_sequence(mad_b4, mylhcbeam=4)

# Apply optics (only for b1b2, b4 will be generated from b1b2)
ost.apply_optics(mad_b1b2, optics_file=configuration['optics_file'])

# Build xsuite collider
collider = pmlhc.build_xsuite_collider(
    sequence_b1=mad_b1b2.sequence.lhcb1,
    sequence_b2=mad_b1b2.sequence.lhcb2,
    sequence_b4=mad_b4.sequence.lhcb2,
    beam_config=configuration['beam_config'],
    enable_imperfections=configuration['enable_imperfections'],
    enable_knob_synthesis=configuration['enable_knob_synthesis'],
    pars_for_imperfections=configuration['pars_for_imperfections'],
    ver_lhc_run=configuration['ver_lhc_run'],
    ver_hllhc_optics=configuration['ver_hllhc_optics'])

with open('collider_00_from_mad.json', 'w') as fid:
    json.dump(collider.to_dict(), fid, cls=xo.JEncoder)



