import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import xtrack as xt


def machine_tuning(line,
        enable_closed_orbit_correction=False,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        dual_pass_tune_and_chroma=False,
        dual_pass_tol_factor: float = 2.,
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
            dual_pass_tol_factor=dual_pass_tol_factor,
            knob_names=knob_names, 
            targets=targets, 
            verbose=verbose
            )


def closed_orbit_correction(line: xt.Line, 
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


def linear_coupling_correction(line: xt.Line,
        knob_names: Dict[str, str], 
        analytical_estimation: bool = False,
        iterative_estimation: int = 0,
        targets: Sequence[str] = None,
        verbose: bool = False,
        limits: Tuple[float, float] = None,
        step: float = 1e-8,
        tol: float = 1e-4,
    ):
    """Perform a correction of linear coupling in the given line.

    Args:
        line (xt.Line): Current line in which to correct linear coupling.
        knob_names (Dict[str, str]): Mapping from the generic knob names to the specific ones of the line.
        analytical_estimation (bool, optional): Perform an analytical estimation of the coupling 
                                                knob values before doing the matching. 
                                                Defaults to False.
        iterative_estimation (int, optional): Try to find a good start condition for the matching by 
                                              iteratively reducing the closest tune approach.
                                              This is the number of steps taken.
                                              Defaults to 0, which means that this estimation is not performed.
        targets (Sequence[str], optional): Mapping of the generic tune targets, to the specific names of the line. 
                                           Only needed if iterative_estimation is not 0. 
                                           Defaults to None.
        verbose (bool, optional): Verbose output of the matching. Defaults to False.
        limits (Sequence[float, float], optional): Limits of the matching, see :class:`xtrack.Vary`. 
                                                   Defaults to None.
        step (float, optional): Step size of the matching, see :class:`xtrack.Vary`. 
                                Defaults to 1e-8.
        tol (float, optional): Tolerance of the matching, see :class:`xtrack.Target`.
                               Defaults to 1e-4.
    """
    assert knob_names is not None
    assert 'c_minus_knob_1' in knob_names
    assert 'c_minus_knob_2' in knob_names
    assert knob_names['c_minus_knob_1'] in line.vars
    assert knob_names['c_minus_knob_2'] in line.vars
    
    if analytical_estimation:
        # Analytically choose a start condition
        analytical_coupling_correction(line, knob_names)

    if iterative_estimation:
        # Iteratively try to find a good start condition.
        iterative_closest_tune_minimization(
            line,
            knob_names=knob_names,
            iteration_steps=iterative_estimation,
            targets=targets,
            verbose=verbose, limits=limits, step=step, tol=tol
        )

    # Match coupling
    print(f'Matching linear coupling')
    line.match(verbose=verbose,
        vary=[
            xt.Vary(name=knob_names['c_minus_knob_1'],
                    limits=limits, step=step),
            xt.Vary(name=knob_names['c_minus_knob_2'],
                    limits=limits, step=step)],
        targets=[xt.Target('c_minus', value=0, tol=tol)])



def analytical_coupling_correction(line: xt.Line, knob_names: Dict[str, str]):
    """Performs an analytical esitmation of the coupling correction, 
    based on the c_minus value from the twiss of the given line.
    This can be useful to have tight limits in :func:`xmask.tuning.linear_coupling_correction`
    even when the original machine coupling is outside of these limits.

    Args:
        line (xt.Line): Line to correct coupling on.
        knob_names(Dict[str, str]): Mapping of the generic coupling knob names to the specific ones of the line.
    """
    assert knob_names is not None
    assert 'c_minus_knob_1' in knob_names
    assert 'c_minus_knob_2' in knob_names
    assert knob_names['c_minus_knob_1'] in line.vars
    assert knob_names['c_minus_knob_2'] in line.vars

    print(f'Estimating linear coupling anlytically')
    for knob_id in ["c_minus_knob_1", "c_minus_knob_2"]:
        knob_name = knob_names[knob_id]

        knob_value0 = line.vars[knob_name]
        c_minus0 = line.twiss().c_minus

        line.vars[knob_name] = knob_value0 - 0.5 * c_minus0
        c_minus_left = line.twiss().c_minus

        line.vars[knob_name] = knob_value0 + 0.5 * c_minus0
        c_minus_right = line.twiss().c_minus
        
        line.vars[knob_name] = knob_value0 + 0.5 * (c_minus_left**2 - c_minus_right**2) / c_minus0


def iterative_closest_tune_minimization(line: xt.Line,
        knob_names: Dict[str, str], 
        iteration_steps: int = 5,
        targets: Sequence[str] = None,
        verbose: bool = False,
        step: float = 1e-8,
        tol: float = 1e-4,
    ):
    """Stepwise try reducing the closest tune approach via tune and coupling knobs, 
       starting at a big tolerance and continuously reducing it until the desired 
       tolerance is reached.

    Args:
        line (xt.Line): Current line in which to correct linear coupling.
        knob_names (Dict[str, str]): Mapping from the generic knob names to the specific ones of the line.
        iteration_steps (int, optional): Number of steps to reduce the closest tune approach. 
                                         Defaults to 5.
        targets (Sequence[str], optional): Mapping of the generic targets, qx, qy and cminus, 
                                           to the specific names of the line. Defaults to None.
        verbose (bool, optional): Verbose output of the matching. Defaults to False.
        step (float, optional): Step size of the matching, see :class:`xtrack.Vary`. 
                                Defaults to 1e-8.
        tol (float, optional): Tolerance of the matching, see :class:`xtrack.Target`.
                               Defaults to 1e-4.

    """
    assert iteration_steps > 0

    assert knob_names is not None
    assert 'c_minus_knob_1' in knob_names
    assert 'c_minus_knob_2' in knob_names
    assert knob_names['c_minus_knob_1'] in line.vars
    assert knob_names['c_minus_knob_2'] in line.vars

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
    
    for ii in range(iteration_steps):
        print(f'Reducing closest tune approach ({ii+1}/{iteration_steps})')
        current_tol = tol * 10**(iteration_steps-ii-1)  # ends at final tolerance
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
                xt.Vary(name=knob_names['c_minus_knob_1'], step=step),
                xt.Vary(name=knob_names['c_minus_knob_2'], step=step)],
            targets=[
                xt.Target('qx', qx_mid, tol=current_tol),
                xt.Target('qy', qy_mid, tol=current_tol), 
            ])
    
    print('Rematching tune')
    line.match(verbose=verbose,
        vary=[
            xt.Vary(name=knob_names['q_knob_1'], step=step),
            xt.Vary(name=knob_names['q_knob_2'], step=step),
        ],
        targets=[
            xt.Target('qx', targets['qx'], tol=tol),
            xt.Target('qy', targets['qy'], tol=tol), 
        ])


def tune_and_chromaticity_correction(line: xt.Line,
        enable_tune_correction: bool = False,
        enable_chromaticity_correction: bool = False,
        dual_pass_tune_and_chroma: bool = False,
        knob_names: Dict[str, str] = None, 
        targets: Dict[str, str] = None,
        verbose: bool = False,
        step_tune: float = 1e-5,
        tol_tune: float = 1e-4,
        step_chroma: float = 1e-2,
        tol_chroma: float = 5e-2,
        dual_pass_tol_factor: float = 2.,
    ):
    """Correct tune and chroma in the given line.

    Args:
        line (xt.Line): Current line in which to correct linear coupling.
        enable_tune_correction (bool, optional): Enables the tune correction. Defaults to False.
        enable_chromaticity_correction (bool, optional): Enables the chroma correction. Defaults to False.
        dual_pass_tune_and_chroma (bool, optional): Performs individual tune and chroma corrections before 
                                                    the combined correction, but only if both tune and chroma 
                                                    are enabled. Defaults to True.
        knob_names (Dict[str, str]): Mapping from the generic knob names to the specific ones of the line.
        targets (Sequence[str], optional): Mapping of the generic targets, qx, qy and dqx, dqy, 
                                           to the specific names of the line. Defaults to None.
        verbose (bool, optional): Verbose output of the matching. Defaults to False.
        step_tune (float, optional): Step size of the tune matching, see :class:`xtrack.Vary`. 
                                     Defaults to 1e-5.
        tol_tune (float, optional): Tolerance of the tune matching, see :class:`xtrack.Target`.
                                    Defaults to 1e-4.
        step_chroma (float, optional): Step size of the chroma matching, see :class:`xtrack.Vary`. 
                                       Defaults to 1e-2.
        tol_chroma (float, optional): Tolerance of the chroma matching, see :class:`xtrack.Target`.
                                      Defaults to 5e-2.
        dual_pass_tol_factor (float, optional): Tolerance factor for the individual matching. Defaults to 2.
    """
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
        targets_tune_single = [
            xt.Target('qx', targets['qx'], tol=dual_pass_tol_factor*tol_tune),
            xt.Target('qy', targets['qy'], tol=dual_pass_tol_factor*tol_tune)
        ]
        line.match(verbose=verbose, vary=vary_tune, targets=targets_tune_single)
        
        print('Matching chromaticity')
        targets_chroma_single = [
            xt.Target('dqx', targets['dqx'], tol=dual_pass_tol_factor*tol_chroma),
            xt.Target('dqy', targets['dqy'], tol=dual_pass_tol_factor*tol_chroma)
        ]
        line.match(verbose=verbose, vary=vary_chroma, targets=targets_chroma_single)

    print('Matching tune and chromaticity')
    line.match(verbose=verbose, vary=vary_tune+vary_chroma, targets=targets_tune+targets_chroma)
