from pathlib import Path

from cpymad.madx import Madx

import xobjects as xo

import pymaskmx as pm
import pymaskmx.lhc as pmlhc

test_data_dir = Path(__file__).parent.parent / "test_data"

common_objects = {}

def test_hllhc14_0_create_collider():
    # Make mad environment
    pm.make_mad_environment(links={
        'acc-models-lhc': str(test_data_dir / 'hllhc14')})

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")
    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    build_sequence(mad_b1b2, mylhcbeam=1)
    build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    apply_optics(mad_b1b2,
        optics_file="acc-models-lhc/round/opt_round_150_1500_thin.madx")

    # For test we do not rely on executatbles to generate coupling knobs
    mad_b1b2.call(
        "acc-models-lhc/_for_test_cminus_knobs_15cm/MB_corr_setting_b1.mad")
    mad_b4.call(
        "acc-models-lhc/_for_test_cminus_knobs_15cm/MB_corr_setting_b2.mad")

    # Build xsuite collider
    collider = pmlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config={'lhcb1':{'beam_energy_tot': 7000},
                     'lhcb2':{'beam_energy_tot': 7000}},
        enable_imperfections=False,
        enable_knob_synthesis=False,
        pars_for_imperfections={},
        ver_lhc_run=None,
        ver_hllhc_optics=1.4)

    assert len(collider.lines.keys()) == 4

    common_objects['collider'] = collider

def test_hllhc14_1_install_beambeam():

    collider = common_objects['collider']
    assert collider is not None

    collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    num_long_range_encounters_per_side=[25, 20, 25, 20],
    num_slices_head_on=11,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    sigmaz=0.076)

    common_objects['collider'] = collider

def test_hllhc14_2_tuning():

    collider = common_objects['collider']
    assert collider is not None



def build_sequence(mad, mylhcbeam, **kwargs):

    # Select beam
    mad.input(f'mylhcbeam = {mylhcbeam}')

    mad.input(

      f'''
      ! Get the toolkit
      call,file=
        "acc-models-lhc/toolkit/macro.madx";
      '''
      '''
      ! Build sequence
      option, -echo,-warn,-info;
      if (mylhcbeam==4){
        call,file="acc-models-lhc/../runIII/lhcb4.seq";
      } else {
        call,file="acc-models-lhc/../runIII/lhc.seq";
      };
      option, -echo, warn,-info;
      '''
      f'''
      !Install HL-LHC
      call, file=
        "acc-models-lhc/hllhc_sequence.madx";
      '''
      '''
      ! Slice nominal sequence
      exec, myslice;
      ''')

    pmlhc.install_errors_placeholders_hllhc(mad)

    mad.input(
      '''
      !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
      if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
      };
      seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;

      ! Install crab cavities (they are off)
      call, file='acc-models-lhc/toolkit/enable_crabcavities.madx';
      on_crab1 = 0;
      on_crab5 = 0;

      ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
      exec, twiss_opt;


      '''
    )


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input('on_alice := on_alice_normalized * 7000./nrj;')
    mad.input('on_lhcb := on_lhcb_normalized * 7000./nrj;')