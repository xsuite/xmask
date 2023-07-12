import json
from pathlib import Path
from xmask.lhc import correct_ir_rdts

import xtrack as xt

def machine_tuning(line,
        enable_closed_orbit_correction=False,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        knob_names=None,
        targets=None,
        line_co_ref=None, co_corr_config=None,
        enable_ir_rdt_correction=False, ir_rdt_corr_config=None,
        verbose=False):

    # Correct closed orbit
    if enable_closed_orbit_correction:
        print(f'Correcting closed orbit')
        assert line_co_ref is not None
        assert co_corr_config is not None
        if isinstance(co_corr_config, (str, Path)):
            with open(co_corr_config, 'r') as fid:
                co_corr_config = json.load(fid)

        line.correct_closed_orbit(
                                reference=line_co_ref,
                                correction_config=co_corr_config)

    if enable_linear_coupling_correction:
        assert knob_names is not None
        assert 'c_minus_knob_1' in knob_names
        assert 'c_minus_knob_2' in knob_names
        assert knob_names['c_minus_knob_1'] in line.vars
        assert knob_names['c_minus_knob_2'] in line.vars
        # Match coupling
        print(f'Matching linear coupling')
        line.match(verbose=verbose,
            vary=[
                xt.Vary(name=knob_names['c_minus_knob_1'],
                        limits=[-0.5e-2, 0.5e-2], step=1e-5),
                xt.Vary(name=knob_names['c_minus_knob_2'],
                        limits=[-0.5e-2, 0.5e-2], step=1e-5)],
            targets=[xt.Target('c_minus', 0, tol=1e-4)])
    
    # Correct nonlinear errors in the IRs
    if enable_ir_rdt_correction:
        print('Correcting nonlinear errors in the IRs')
        if isinstance(ir_rdt_corr_config, (str, Path)):
            with open(ir_rdt_corr_config, 'r') as fid:
                ir_rdt_corr_config = json.load(fid)
        
        line_name = ir_rdt_corr_config.pop('line_name')
        if "output" in ir_rdt_corr_config:
            ir_rdt_corr_config["output"] = ir_rdt_corr_config["output"].format(line_name)

        irnl_correction = correct_ir_rdts.calculate_correction(
            line, 
            beams=[int(line_name[-1])],
            **ir_rdt_corr_config,
        )
        correct_ir_rdts.apply_correction(line, correction=irnl_correction)
        

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

            vary.append(xt.Vary(knob_names['q_knob_1'], step=1e-5))
            vary.append(xt.Vary(knob_names['q_knob_2'], step=1e-5))
            match_targets.append(xt.Target('qx', targets['qx'], tol=1e-4))
            match_targets.append(xt.Target('qy', targets['qy'], tol=1e-4))

        if enable_chromaticity_correction:
            assert knob_names is not None
            assert 'dq_knob_1' in knob_names
            assert 'dq_knob_2' in knob_names
            assert targets is not None
            assert 'dqx' in targets
            assert 'dqy' in targets

            vary.append(xt.Vary(knob_names['dq_knob_1'], step=1e-2))
            vary.append(xt.Vary(knob_names['dq_knob_2'], step=1e-2))
            match_targets.append(xt.Target('dqx', targets['dqx'], tol=0.05))
            match_targets.append(xt.Target('dqy', targets['dqy'], tol=0.05))

        print(f'Matching tune and chromaticity')
        line.match(verbose=verbose, vary=vary, targets=match_targets)
