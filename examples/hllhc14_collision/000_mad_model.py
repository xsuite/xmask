import json
import yaml

from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
import xpart as xp

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
mad = Madx(command_log="mad_collider.log")
mad.globals.par_verbose = int(configuration['verbose_mad_parts'])

# Build sequence, load optics, define beam
ost.build_sequence(mad, beam=1, optics_version=configuration['optics_version'])
ost.apply_optics(mad, optics_file=configuration['optics_file'])

beam_config = configuration['beam_config']
pm.attach_beam_to_sequence(mad.sequence.lhcb1, beam_to_configure=1,
                            beam_configuration=beam_config['lhcb1'])
pm.attach_beam_to_sequence(mad.sequence.lhcb2, beam_to_configure=2,
                            beam_configuration=beam_config['lhcb2'])

# Warm up (seems I need to twiss for mad to load everything)
mad.use('lhcb1'); mad.twiss(); mad.use('lhcb2'); mad.twiss()

# Generate beam 4
mad_b4 = Madx(command_log="mad_b4.log")
ost.build_sequence(mad_b4, beam=4, optics_version=configuration['optics_version'])
pm.configure_b4_from_b2(
    sequence_b4=mad_b4.sequence.lhcb2,
    sequence_b2=mad.sequence.lhcb2)

# Save lines for closed orbit reference
lines_co_ref = pm.save_lines_for_closed_orbit_reference(
    sequence_clockwise=mad.sequence.lhcb1,
    sequence_anticlockwise=mad_b4.sequence.lhcb2)

lines_to_track = {}
for sequence_to_track, mad_track in zip(['lhcb1', 'lhcb2'], [mad, mad_b4]):

    # Final use
    mad_track.use(sequence_to_track)

    # We work exclusively on the flat machine
    mad_track.input('exec, crossing_disable;')
    mad_track.input('exec, crossing_save;') # In this way crossing_restore keeps the flat machine

    # Install and correct errors
    pmlhc.install_correct_errors_and_synthesisize_knobs(mad_track,
        enable_imperfections=configuration['enable_imperfections'],
        enable_knob_synthesis= configuration['enable_knob_synthesis'],
        pars_for_imperfections=configuration['pars_for_imperfections'])

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
