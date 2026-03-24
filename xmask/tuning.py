import json
from pathlib import Path

import xtrack as xt
import xmask as xm

def machine_tuning(line,
        enable_closed_orbit_correction=False,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        knob_names=None,
        targets=None,
        step_q_knob=None, step_dq_knob=None, step_c_minus_knob=None,
        tol_tune=None, tol_chromaticity=None, tol_c_minus=None,
        line_co_ref=None, co_corr_config=None,
        verbose=False):

    if step_q_knob is None:
        step_q_knob = 1e-5
    if step_dq_knob is None:
        step_dq_knob = 1e-2
    if step_c_minus_knob is None:
        step_c_minus_knob = 1e-5

    if tol_tune is None:
        tol_tune = 1e-4
    if tol_chromaticity is None:
        tol_chromaticity = 0.05
    if tol_c_minus is None:
        tol_c_minus = 2e-4

    # Correct closed orbit
    if enable_closed_orbit_correction:
        print()
        print(f'Correcting closed orbit')
        assert line_co_ref is not None
        assert co_corr_config is not None
        if isinstance(co_corr_config, (str, Path)):
            with open(co_corr_config, 'r') as fid:
                co_corr_config = json.load(fid)

        if line_co_ref.env is not line.env:
            xm.transfer_vars_to_env(source=line, dest=line_co_ref)

        line._xmask_correct_closed_orbit(
                                reference=line_co_ref,
                                correction_config=co_corr_config)

    if enable_linear_coupling_correction:
        assert knob_names is not None
        assert 'c_minus_knob_1' in knob_names
        assert 'c_minus_knob_2' in knob_names
        # Match coupling
        print()
        print(f'Matching linear coupling')
        line.match(verbose=verbose,
            compute_chromatic_properties=False,
            vary=[
                xt.Vary(name=knob_names['c_minus_knob_1'],
                        limits=[-0.5e-2, 0.5e-2], step=step_c_minus_knob),
                xt.Vary(name=knob_names['c_minus_knob_2'],
                        limits=[-0.5e-2, 0.5e-2], step=step_c_minus_knob)],
            targets=[xt.Target('c_minus', 0, tol=tol_c_minus)])

    # Match tune and chromaticity
    if enable_tune_correction or enable_chromaticity_correction:

        vary = []
        match_targets = []

        if enable_tune_correction:
            assert knob_names is not None
            assert 'q_knob_1' in knob_names
            assert 'q_knob_2' in knob_names
            assert targets is not None
            assert 'qx' in targets
            assert 'qy' in targets

            vary.append(xt.Vary(knob_names['q_knob_1'], step=step_q_knob))
            vary.append(xt.Vary(knob_names['q_knob_2'], step=step_q_knob))
            match_targets.append(xt.Target('qx', targets['qx'], tol=tol_tune))
            match_targets.append(xt.Target('qy', targets['qy'], tol=tol_tune))

        if enable_chromaticity_correction:
            assert knob_names is not None
            assert 'dq_knob_1' in knob_names
            assert 'dq_knob_2' in knob_names
            assert targets is not None
            assert 'dqx' in targets
            assert 'dqy' in targets

            vary.append(xt.Vary(knob_names['dq_knob_1'], step=step_dq_knob))
            vary.append(xt.Vary(knob_names['dq_knob_2'], step=step_dq_knob))
            match_targets.append(xt.Target('dqx', targets['dqx'], tol=tol_chromaticity))
            match_targets.append(xt.Target('dqy', targets['dqy'], tol=tol_chromaticity))

        print()
        print(f'Matching tune and chromaticity')
        line.match(verbose=verbose, vary=vary, targets=match_targets)

def transfer_vars_to_env(source, dest):
    old_default_to_zero = dest.vars.default_to_zero
    dest.vars.default_to_zero = True
    source_dct = source.vars.get_table(compact=False).to_dict()
    for nn, vv in source_dct.items():
        if isinstance(vv, str):
            dest.ref[nn] = eval(vv, locals=dest.ref_manager.containers)
        else:
            dest.ref[nn] = vv
    dest.vars.default_to_zero = old_default_to_zero
