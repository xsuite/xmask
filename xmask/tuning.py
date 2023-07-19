import json
from pathlib import Path

import xtrack as xt

def machine_tuning(line,
        enable_closed_orbit_correction=False,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        dual_pass_tune_and_chroma=True,
        knob_names=None, 
        targets=None,
        coupling_correction_analytical_estimation=False,
        coupling_correction_iterative_estimation=0,
        line_co_ref=None, co_corr_config=None,
        verbose=False):

    if enable_closed_orbit_correction:
        closed_orbit_correction(line, 
            line_co_ref=line_co_ref, 
            co_corr_config=co_corr_config
        )

    if enable_linear_coupling_correction:
        linear_coupling_correction(line,
            knob_names=knob_names,
            analytical_estimation=coupling_correction_analytical_estimation,
            iterative_estimation=coupling_correction_iterative_estimation,
            targets=targets,
            verbose=verbose,
        )

    if enable_tune_correction or enable_chromaticity_correction:
        tune_and_chromaticity_correction(line, 
            enable_tune_correction=enable_tune_correction, 
            enable_chromaticity_correction=enable_chromaticity_correction,
            dual_pass_tune_and_chroma=dual_pass_tune_and_chroma,
            knob_names=knob_names, 
            targets=targets, 
            verbose=verbose
            )


def closed_orbit_correction(line, 
    line_co_ref=None, 
    co_corr_config=None
    ):
    print(f'Correcting closed orbit')
    assert line_co_ref is not None
    assert co_corr_config is not None
    if isinstance(co_corr_config, (str, Path)):
        with open(co_corr_config, 'r') as fid:
            co_corr_config = json.load(fid)

    line.correct_closed_orbit(
        reference=line_co_ref,
        correction_config=co_corr_config)


def linear_coupling_correction(line,
        knob_names=None, 
        analytical_estimation=False,
        iterative_estimation=0,
        targets=None,
        verbose=False,
    ):
    assert knob_names is not None
    assert 'c_minus_knob_1' in knob_names
    assert 'c_minus_knob_2' in knob_names
    assert knob_names['c_minus_knob_1'] in line.vars
    assert knob_names['c_minus_knob_2'] in line.vars
    
    limits = [-0.5e-2, 0.5e-2]
    step = 1e-8
    tolerance=1e-4

    if analytical_estimation:
        # Analytically choose a start condition
        print(f'Estimating linear coupling')
        for knob_id in ["c_minus_knob_1", "c_minus_knob_2"]:
            knob_name = knob_names[knob_id]

            knob_value0 = line.vars[knob_name]
            c_minus0 = line.twiss().c_minus

            line.vars[knob_name] = knob_value0 - 0.5 * c_minus0
            c_minus_left = line.twiss().c_minus

            line.vars[knob_name] = knob_value0 + 0.5 * c_minus0
            c_minus_right = line.twiss().c_minus
            
            line.vars[knob_name] = knob_value0 + 0.5 * (c_minus_left**2 - c_minus_right**2) / c_minus0

    if iterative_estimation:
        # Iteratively try to find a good start condition 
        assert 'q_knob_1' in knob_names
        assert 'q_knob_2' in knob_names
        assert knob_names['q_knob_1'] in line.vars
        assert knob_names['q_knob_2'] in line.vars
        assert targets is not None
        assert 'qx' in targets
        assert 'qy' in targets

        qx_frac, qy_frac = targets['qx'] % 1, targets['qy'] % 1
        qmid_frac = 0.5 * (qx_frac + qy_frac)
        qx_mid = int(targets['qx']) + qmid_frac
        qy_mid = int(targets['qy']) + qmid_frac
        
        for ii in range(iterative_estimation):
            print(f'Matching linear coupling ({ii+1}/{iterative_estimation})')
            current_tol = tolerance * 10**(iterative_estimation-ii-1)  # ends at final tolerance
            line.match(verbose=verbose,
                vary=[
                    xt.Vary(name=knob_names['q_knob_1'], step=step),
                    xt.Vary(name=knob_names['q_knob_2'], step=step),
                ],
                targets=[
                    xt.Target('qx', qx_mid, tol=current_tol/2),
                    xt.Target('qy', qy_mid, tol=current_tol/2), 
                ])
            line.match(verbose=verbose,
                vary=[
                    xt.Vary(name=knob_names['c_minus_knob_1'],
                            limits=limits, step=step),
                    xt.Vary(name=knob_names['c_minus_knob_2'],
                            limits=limits, step=step)],
                targets=[xt.Target('c_minus', value=0, tol=current_tol)])
        
        print('Rematching tune')
        line.match(verbose=verbose,
            vary=[
                xt.Vary(name=knob_names['q_knob_1'], step=step),
                xt.Vary(name=knob_names['q_knob_2'], step=step),
            ],
            targets=[
                xt.Target('qx', targets['qx'], tol=tolerance),
                xt.Target('qy', targets['qy'], tol=tolerance), 
            ])

    # Match coupling
    print(f'Matching linear coupling')
    line.match(verbose=verbose,
        vary=[
            xt.Vary(name=knob_names['c_minus_knob_1'],
                    limits=limits, step=step),
            xt.Vary(name=knob_names['c_minus_knob_2'],
                    limits=limits, step=step)],
        targets=[xt.Target('c_minus', value=0, tol=tolerance)])


def tune_and_chromaticity_correction(line,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        dual_pass_tune_and_chroma=True,
        knob_names=None, 
        targets=None,
        verbose=False,
        step_tune=1e-5,
        tol_tune=1e-4,
        step_chroma=1e-2,
        tol_chroma=5e-2,
    ):
    vary_tune, vary_chroma = [], []
    targets_tune, targets_chroma = [], []

    if enable_tune_correction:
        assert knob_names is not None
        assert 'q_knob_1' in knob_names
        assert 'q_knob_2' in knob_names
        assert knob_names['q_knob_1'] in line.vars
        assert knob_names['q_knob_2'] in line.vars
        assert targets is not None
        assert 'qx' in targets
        assert 'qy' in targets

        vary_tune = [
            xt.Vary(knob_names['q_knob_1'], step=step_tune),
            xt.Vary(knob_names['q_knob_2'], step=step_tune)
        ]
        targets_tune = [
            xt.Target('qx', targets['qx'], tol=tol_tune),
            xt.Target('qy', targets['qy'], tol=tol_tune)
        ]

    if enable_chromaticity_correction:
        assert knob_names is not None
        assert 'dq_knob_1' in knob_names
        assert 'dq_knob_2' in knob_names
        assert knob_names['dq_knob_1'] in line.vars
        assert knob_names['dq_knob_2'] in line.vars
        assert targets is not None
        assert 'dqx' in targets
        assert 'dqy' in targets

        vary_chroma = [
            xt.Vary(knob_names['dq_knob_1'], step=step_chroma),
            xt.Vary(knob_names['dq_knob_2'], step=step_chroma)
        ]
        targets_chroma = [
            xt.Target('dqx', targets['dqx'], tol=tol_chroma),
            xt.Target('dqy', targets['dqy'], tol=tol_chroma)
        ]
    
    if dual_pass_tune_and_chroma and enable_tune_correction and enable_chromaticity_correction:
        print('Matching tune')
        line.match(verbose=verbose, vary=vary_tune, targets=targets_tune)

        print('Matching chromaticity')
        line.match(verbose=verbose, vary=vary_chroma, targets=targets_chroma)

    print('Matching tune and chromaticity')
    line.match(verbose=verbose, vary=vary_tune+vary_chroma, targets=targets_tune+targets_chroma)
