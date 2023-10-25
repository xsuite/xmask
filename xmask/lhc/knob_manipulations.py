from pathlib import Path
from typing import Dict, Sequence
import numpy as np
from scipy.constants import c as clight
import tfs
import xtrack as xt


def rename_coupling_knobs_and_coefficients(line, beamn):

    line.vars[f'c_minus_re_b{beamn}'] = 0
    line.vars[f'c_minus_im_b{beamn}'] = 0
    for ii in [1, 2, 3, 4, 5, 6, 7, 8]:
        for jj, nn in zip([1, 2], ['re', 'im']):
            old_name = f'b{ii}{jj}'
            new_name = f'coeff_skew_{ii}{jj}_b{beamn}'

            # Copy value in new variable
            line.vars[new_name] = line.vars[old_name]._value

            # Zero old variable
            line.vars[old_name] = 0

            # Identify controlled circuit
            targets = line.vars[old_name]._find_dependant_targets()
            if len(targets) > 1: # Controls something
                ttt = [t for t in targets if repr(t).startswith('vars[') and
                    repr(t) != f"vars['{old_name}']"]
                assert len(ttt) > 0
                assert len(ttt) < 3

                for kqs_knob in ttt:
                    kqs_knob_str = repr(kqs_knob)
                    assert "'" in kqs_knob_str
                    assert '"' not in kqs_knob_str
                    var_name = kqs_knob_str.split("'")[1]
                    assert var_name.startswith('kqs')
                    line.vars[var_name] += (line.vars[new_name]
                                * line.vars[f'c_minus_{nn}_b{beamn}'])


def define_octupole_current_knobs(line, beamn):

    line.vars[f'p0c_b{beamn}'] = line.particle_ref.p0c[0]
    line.vars[f'q0_b{beamn}'] = line.particle_ref.q0
    line.vars[f'brho0_b{beamn}'] = (line.vars[f'p0c_b{beamn}']
                                / line.vars[f'q0_b{beamn}'] / clight)

    line.vars[f'i_oct_b{beamn}'] = 0
    for ss in '12 23 34 45 56 67 78 81'.split():
        line.vars[f'kof.a{ss}b{beamn}'] = (
            line.vars['kmax_mo']
            * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
            / line.vars[f'brho0_b{beamn}'])
        line.vars[f'kod.a{ss}b{beamn}'] = (
            line.vars['kmax_mo']
            * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
            / line.vars[f'brho0_b{beamn}'])


def add_correction_term_to_dipole_correctors(line):
    # Add correction term to all dipole correctors
    line.vars['on_corr_co'] = 1
    for kk in list(line.vars.keys()):
        if kk.startswith('acb'):
            line.vars['corr_co_'+kk] = 0
            line.vars[kk] += (line.vars['corr_co_'+kk] * line.vars['on_corr_co'])


def calculate_coupling_coefficients_per_sector(
        df: tfs.TfsDataFrame, 
        deactivate_sectors: Sequence[str] = ('12', '45', '56', '81')
    ) -> tfs.TfsDataFrame:
    """ Calculate the coupling knob coefficients as in corr_MB_ats_v4.
    This is basically Eq. (59) in https://cds.cern.ch/record/522049/files/lhc-project-report-501.pdf ,
    with cosine for the real part and sine for the imaginary part.

    What is happening here is, that we are building a matrix for the equation-system

    M * [MQS12,..., MQS81] = -[RE, IM]

    where RE and IM are the real and imaginary parts of the coupling coefficients, 
    and therefore the knobs we want to create.
    MQS12 - MQS81 is the total powering of the MQS per arc, which we steer with 
    the knob. All MQS are assumed to be powered equally.

    After building the matrix, we "solve" the equation system via pseudo-inverese M+
    and get therefore the definition of the coupling knob.

    [MQS12,..., MQS81] = - M+ * [RE, IM]

    HINT: The MINN function in corr_MB_ats_v4 is simply the calculation the pseudo-inverse:
    M+  = M' (M * M')^-1  (' = transpose, ^-1 = inverse)
    including the minus on the rhs of the equation.


    TODO:
        - Get beta and phases from the beamline directly (instead of the TFS dataframe)
        - Make the number of MQS more flexible, could maybe be parsed from the beamline
        - Why is there an absolute value in Eq. (59) but not in the code?
        - Add a2 correction to the KQS definition (as in corr_MB_ats_v4)
        - Use actual fractional tune split of the current machine (see also https://cds.cern.ch/record/2778887/files/CERN-ACC-NOTE-2021-0022.pdf)
        - Calculate also the contribution to f_1010 and try to set to zero
        - Explain why some sectors are deactivated? Reference?

    Args:
        df (tfs.TfsDataFrame): Dataframe containing the optics of the machine. 

        beam (int): _description_
        deactivate_sectors (bool, optional): _description_. Defaults to True.
    """
    BETX, BETY, MUX, MUY = "BETX", "BETY", "MUX", "MUY"
    MQS_PER_SECTOR = 4

    sectors = '12 23 34 45 56 67 78 81'.split()
    
    mqs_sectors = [fr"MQS.*(R{i}|L{i+1}).B" for i in range(1, 9)]
    m = np.ndarray([2, len(mqs_sectors)])

    for isector, mqs_regex in enumerate(mqs_sectors):
        sector_slices = df.index.str.match(mqs_regex)
        df_sec = df.loc[sector_slices]
        coeff = MQS_PER_SECTOR / len(sector_slices) * np.sqrt(df_sec[BETX] * df_sec[BETY])
        phase = 2*np.pi * (df_sec[MUX] - df_sec[MUY])

        for idx, fun in enumerate((np.cos, np.sin)):
            m[idx, isector] =  (coeff * fun(phase)).sum() 
        
    m = m * 0.32  / (2 * np.pi)  # I think this is the tune split?

    if deactivate_sectors:
        mask = [s in deactivate_sectors for s in sectors]
        m[: , mask] = 0

    result = tfs.TfsDataFrame(
        data=-np.linalg.pinv(m),
        index=sectors, 
        columns=["re", "im"]
    )

    # remove numerical noise:
    result = result.where(result.abs() > 1e-15, 0)

    return result


def create_coupling_knobs(line: xt.Line, beamn: int, optics: Path = Path("temp/optics0_MB.mad")):
    """ Create coupling knobs in the beam-line.
    WARNING: This function will not take a2 errors into account! 
    Normally, the a2 errors are also corrected with the MQS, but in this function 
    the MQS powering is fully controlled by the coupling knobs. 
    See also the todo's in :func:`xmask.lhc.knob_manipulations.calculate_coupling_coefficients_per_sector`

    Args:
        line (xt.Line): Line to incorporate the knobs in. 
        beamn (int): Beam number associated with the line.
        optics (Path, optional): Path to the TFS-File containing the optics to be used. 
                                 Defaults to Path("temp/optics0_MB.mad").
    """
    print(f"\n Creating Coupling Knobs for beam {beamn} ---")
    SINGLE_KQS_LISTS = {
    # Sectors with commonly powered MQSs
        1: ['23', '45', '67', '81'],
        2: ['12', '34', '56', '78'], 
    }

    if beamn == 4:
        beamn = 2  # same behaviour in this case

    beam_sign = 1 if beamn == 1 else -1

    df = tfs.read(optics, index="NAME")
    df = beam_sign * calculate_coupling_coefficients_per_sector(df)

    knob_name_real =  f'c_minus_re_b{beamn}'
    knob_name_imag =  f'c_minus_im_b{beamn}'


    line.vars[knob_name_real] = 0
    line.vars[knob_name_imag] = 0

    knob_re = line.vars[knob_name_real]
    knob_im = line.vars[knob_name_imag]

    for idx_sector, sector in enumerate(df.index, start=1):
        # Better naming according to jdilly:
        # coeff_name_real = f"coeff_skew_re_arc{sector}_b{beamn}"
        # coeff_name_imag = f"coeff_skew_im_arc{sector}_b{beamn}"

        # Going with renaming scheme from above:
        coeff_name_real = f"coeff_skew_{idx_sector}{1}_b{beamn}"
        coeff_name_imag = f"coeff_skew_{idx_sector}{2}_b{beamn}"

        line.vars[coeff_name_real] = df.loc[sector, "re"]
        line.vars[coeff_name_imag] = df.loc[sector, "im"]

        coeff_real = line.vars[coeff_name_real]
        coeff_imag = line.vars[coeff_name_imag]

        if sector in SINGLE_KQS_LISTS[beamn]:
            line.vars[f"kqs.a{sector}b{beamn}"] = coeff_real * knob_re + coeff_imag * knob_im
        else:
            line.vars[f"kqs.r{sector[0]}b{beamn}"] = coeff_real * knob_re + coeff_imag * knob_im
            line.vars[f"kqs.l{sector[1]}b{beamn}"] = coeff_real * knob_re + coeff_imag * knob_im

    