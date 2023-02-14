
import os

def install_correct_errors_and_synthesisize_knobs(mad_track, enable_imperfections,
                        enable_knob_synthesis, pars_for_imperfections,
                        ver_lhc_run=None, ver_hllhc_optics=None):

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
        mad_track.input(f'call, file="{scripts_folder}/submodule_04e_s1_synthesize_knobs.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04e_correction.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04f_final.madx";')

        mad_track.input('exec, crossing_restore;')
    else:
        # Synthesize knobs
        if enable_knob_synthesis:
            mad_track.input(f'call, file="{scripts_folder}/submodule_04a_s1_prepare_nom_twiss_table.madx";')
            mad_track.input(f'call, file="{scripts_folder}/submodule_04e_s1_synthesize_knobs.madx";')