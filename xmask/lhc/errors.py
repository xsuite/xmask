
import os
import pathlib

lhc_module_folder = pathlib.Path(__file__).parent

def install_errors_placeholders_hllhc(mad):
    os.system(f'rm errors')
    os.symlink(lhc_module_folder / 'lhcerrors', 'errors')

    mad.input('''
      ! Install placeholder elements for errors (set to zero)
      call, file="errors/HL-LHC/install_MQXF_fringenl.madx";  ! adding fringe place holder
      call, file="errors/HL-LHC/install_MCBXFAB_errors.madx"; !adding D1 corrector placeholders in IR1/5 (for errors)
      call, file="errors/HL-LHC/install_MCBRD_errors.madx";   ! adding D2 corrector placeholders in IR1/5 (for errors)
      call, file="errors/HL-LHC/install_NLC_errors.madx";     !adding non-linear corrector placeholders in IR1/5 (for errors)
    ''')

def install_correct_errors_and_synthesisize_knobs(mad_track, enable_imperfections,
                        pars_for_imperfections,
                        enable_legacy_mb_corrections, 
                        enable_legacy_nl_corrections,
                        ver_lhc_run=None, ver_hllhc_optics=None):

    os.system(f'rm errors')
    os.symlink(lhc_module_folder / 'lhcerrors', 'errors')

    assert ver_lhc_run is not None or ver_hllhc_optics is not None, (
        'Must specify either ver_lhc_run or ver_hllhc_optics')

    if ver_lhc_run is not None:
        assert ver_hllhc_optics is None, (
            'Must specify either ver_lhc_run or ver_hllhc_optics, not both')
        assert type(ver_lhc_run) is float
        mad_track.globals.ver_lhc_run = ver_lhc_run

    if ver_hllhc_optics is not None:
        assert ver_lhc_run is None, (
            'Must specify either ver_lhc_run or ver_hllhc_optics, not both')
        assert type(ver_hllhc_optics) is float
        mad_track.globals.ver_hllhc_optics = ver_hllhc_optics

    mad_track.globals.par_verbose = 1

    # Force on_disp = 0
    mad_track.globals.on_disp = 0. # will be restored later

    scripts_folder = os.path.dirname(__file__) + '/madx_scripts'

    # Install and correct errors
    if enable_imperfections:
        for kk, vv in pars_for_imperfections.items():
            mad_track.globals[kk] = vv
        mad_track.input(f'call, file="{scripts_folder}/submodule_04a_preparation.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04a_s1_prepare_nom_twiss_table.madx";')

        mad_track.input('exec, crossing_disable;')

        mad_track.input(f'call, file="{scripts_folder}/submodule_04b_alignsep.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04c_errortables.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04d_efcomp.madx";')

        if enable_legacy_mb_corrections:
            # also synthesizes coupling knobs
            mad_track.input(f'call, file="{scripts_folder}/submodule_04e_mb_arc_correction_and_coupling_knobs.madx";')

        if enable_legacy_nl_corrections:
            mad_track.input(f'call, file="{scripts_folder}/submodule_04e_ir_nl_correction.madx";')

        mad_track.input(f'call, file="{scripts_folder}/submodule_04f_final.madx";')

        mad_track.input('exec, crossing_restore;')
    else:
        # Nominal twiss tables needed currently for python knob-synthesis
        mad_track.input(f'call, file="{scripts_folder}/submodule_04a_s1_prepare_nom_twiss_table.madx";')