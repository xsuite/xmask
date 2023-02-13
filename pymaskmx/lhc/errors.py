
import os

def install_correct_errors_and_synthesisize_knobs(mad_track, enable_imperfections,
                        enable_knob_synthesis, pars_for_imperfections):
    # Force on_disp = 0
    mad_track.globals.on_disp = 0. # will be restored later

    scripts_folder = os.path.dirname(__file__) + '/madx_scripts'

    # Install and correct errors
    if enable_imperfections:
        for kk, vv in pars_for_imperfections.items():
            mad_track.globals[kk] = vv
        mad_track.input(f'call, file="{scripts_folder}/submodule_04a_preparation.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04b_alignsep.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04c_errortables.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04d_efcomp.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04e_correction.madx";')
        mad_track.input(f'call, file="{scripts_folder}/submodule_04f_final.madx";')
    else:
        # Synthesize knobs
        if enable_knob_synthesis:
            mad_track.input(f'call, file="{scripts_folder}/submodule_04a_s1_prepare_nom_twiss_table.madx";')
            mad_track.input(f'call, file="{scripts_folder}/submodule_04e_s1_synthesize_knobs.madx";')