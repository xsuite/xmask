import json
import yaml

from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
import xpart as xp

import pymaskmx as pm
import pymaskmx.lhc as pmlhc

# Assumptions from the madx model and optics
# - Machine energy is stored in madx variable "nrj" (in GeV)
# - The variable "mylhcbeam" is set to one in the madx instance holding
#   b1 and b2 (both clockwise), and to 4 in the madx instance holding b4
#   (anti-clockwise).
# - Version of optics is stored in madx variable "ver_lhc_run" for LHC and
#   "ver_hllhc_optics" for HL-LHC
# - Macros are available to save/desable/load orbit bumps, which are called
#   "crossing_save", "crossing_disable", and "crossing_restore".

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

# Beam definitions (only for b1b2, b4 will be generated from b1b2)
beam_config = configuration['beam_config']
pm.attach_beam_to_sequence(mad_b1b2.sequence.lhcb1, beam_to_configure=1,
                            beam_configuration=beam_config['lhcb1'])
pm.attach_beam_to_sequence(mad_b1b2.sequence.lhcb2, beam_to_configure=2,
                            beam_configuration=beam_config['lhcb2'])

# Warm up (seems I need to twiss for mad to load everything)
mad_b1b2.use('lhcb1'); mad_b1b2.twiss(); mad_b1b2.use('lhcb2'); mad_b1b2.twiss()

# Generate beam 4
pm.configure_b4_from_b2(
    sequence_b4=mad_b4.sequence.lhcb2,
    sequence_b2=mad_b1b2.sequence.lhcb2)

# Save lines for closed orbit reference
lines_co_ref = pm.save_lines_for_closed_orbit_reference(
    sequence_clockwise=mad_b1b2.sequence.lhcb1,
    sequence_anticlockwise=mad_b4.sequence.lhcb2)

lines_to_track = {}
for sequence_to_track, mad_track in zip(['lhcb1', 'lhcb2'], [mad_b1b2, mad_b4]):

    # Final use
    mad_track.use(sequence_to_track)

    # We work exclusively on the flat machine
    mad_track.input('exec, crossing_disable;')
    mad_track.input('exec, crossing_save;') # In this way crossing_restore keeps the flat machine

    # Install and correct errors
    pmlhc.install_correct_errors_and_synthesisize_knobs(mad_track,
        enable_imperfections=configuration['enable_imperfections'],
        enable_knob_synthesis= configuration['enable_knob_synthesis'],
        pars_for_imperfections=configuration['pars_for_imperfections'],
        ver_lhc_run=configuration.get('ver_lhc_run', None),
        ver_hllhc_optics=configuration.get('ver_hllhc_optics', None))

    # Prepare xsuite line
    line = xt.Line.from_madx_sequence(
        mad_track.sequence[sequence_to_track], apply_madx_errors=True,
        deferred_expressions=True,
        replace_in_expr={'bv_aux': 'bvaux_'+sequence_to_track})
    mad_beam = mad_track.sequence[sequence_to_track].beam
    line.particle_ref = xp.Particles(p0c = mad_beam.pc*1e9,
        q0 = mad_beam.charge, mass0 = mad_beam.mass*1e9)

    # Prepare coupling and octupole knobs
    pmlhc.rename_coupling_knobs_and_coefficients(line=line,
                                           beamn=int(sequence_to_track[-1]))
    pmlhc.define_octupole_current_knobs(line=line, beamn=int(sequence_to_track[-1]))
    lines_to_track[sequence_to_track] = line


collider = xt.Multiline(
    lines={
        'lhcb1': lines_to_track['lhcb1'],
        'lhcb2': lines_to_track['lhcb2'],
        'lhcb1_co_ref': lines_co_ref['lhcb1_co_ref'],
        'lhcb2_co_ref': lines_co_ref['lhcb2_co_ref'],
    })

collider['lhcb1_co_ref'].particle_ref = collider['lhcb1'].particle_ref.copy()
collider['lhcb2_co_ref'].particle_ref = collider['lhcb2'].particle_ref.copy()

pmlhc.add_correction_term_to_dipole_correctors(collider)

# Save the two lines to json
with open('collider_00_from_mad.json', 'w') as fid:
    dct = collider.to_dict()
    json.dump(dct, fid, cls=xo.JEncoder)
