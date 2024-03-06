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


def closed_orbit_correction(line: xt.Line, 
    line_co_ref=None, 
    co_corr_config=None,
    verbose: bool = False,
    ):
    print(f'Correcting closed orbit')
    assert line_co_ref is not None
    assert co_corr_config is not None
    if isinstance(co_corr_config, (str, Path)):
        with open(co_corr_config, 'r') as fid:
            co_corr_config = json.load(fid)

    line.correct_closed_orbit(
        reference=line_co_ref,
        correction_config=co_corr_config,
        verbose=verbose,
    )


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


def tune_and_chromaticity_correction(line: xt.Line,
        enable_tune_correction: bool = False,
        enable_chromaticity_correction: bool = False,
        dual_pass_tune_and_chroma: bool = False,
        knob_names: Dict[str, str] = None, 
        targets: Dict[str, str] = None,
        verbose: bool = False,
        step_tune: float = 1e-5,
        tol_tune: float = 1e-4,
        limits_tune: Tuple[float, float] = None,
        step_chroma: float = 1e-2,
        tol_chroma: float = 5e-2,
        limits_chroma: Tuple[float, float]  = None, 
        **kwargs,
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
        limits_tune (Tuple[float, float], optional): Limits of the tune matching, see :class:`xtrack.Vary`.
        step_chroma (float, optional): Step size of the chroma matching, see :class:`xtrack.Vary`. 
                                       Defaults to 1e-2.
        tol_chroma (float, optional): Tolerance of the chroma matching, see :class:`xtrack.Target`.
                                      Defaults to 5e-2.
        limits_chroma (Tuple[float, float], optional): Limits of the chroma matching, see :class:`xtrack.Vary`.
        kwargs: Arguments to be passed to the matching function.
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
            xt.Vary(knob_names['q_knob_1'], step=step_tune, limits=limits_tune),
            xt.Vary(knob_names['q_knob_2'], step=step_tune, limits=limits_tune)
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
            xt.Vary(knob_names['dq_knob_1'], step=step_chroma, limits=limits_chroma),
            xt.Vary(knob_names['dq_knob_2'], step=step_chroma, limits=limits_chroma)
        ]
        targets_chroma = [
            xt.Target('dqx', targets['dqx'], tol=tol_chroma),
            xt.Target('dqy', targets['dqy'], tol=tol_chroma)
        ]
    
    if dual_pass_tune_and_chroma and enable_tune_correction and enable_chromaticity_correction:
        # Correct Tune and Chroma individually first
        print('Matching tune')
        line.match(verbose=verbose, vary=vary_tune, targets=targets_tune, **kwargs)
        
        print('Matching chromaticity')
        line.match(verbose=verbose, vary=vary_chroma, targets=targets_chroma, **kwargs)

    print('Matching tune and chromaticity')
    line.match(verbose=verbose, vary=vary_tune+vary_chroma, targets=targets_tune+targets_chroma, **kwargs)
