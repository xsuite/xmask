import xtrack as xt
from math import factorial

from load_wise import set_multipole_errors_in_line

import numpy as np

# | Position | Q1-A      | Q1-B      | Q3-A      | Q3-B      |
# |----------|-----------|-----------|-----------|-----------|
# | l5       | MQXFA10   | MQXFA11   | MQXFA21   | MQXFA22   |
# | r1       | MQXFA07b  | MQXFA15   | MQXFA14b  | MQXFA08b  |
# | r5       | MQXFA19   | MQXFA20   | MQXFA23   | MQXFA24   |
# | l1       | MQXFA18   | MQXFA13b  | MQXFA17b  | MQXFA12b  |

# For the Q2
# prefix: "mqxfb"
# slice_ids: ["a2", "b2"]
# assignment:
#   l5: ["MQXFB05", "MQXFB04"]
#   r1: ["MQXFB03", "MQXFB06"]
#   r5: ["MQXFB07", "MQXFB08"]
#   l1: ["MQXFB11", "MQXFB10"]

# To be checked:

magnet_asset_association = {
    'mqxfa.b3l1': 'mqxfa12b',
    'mqxfa.a3l1': 'mqxfa17b',
    'mqxfb.b2l1': 'mqxfb10',
    'mqxfb.a2l1': 'mqxfb11',
    'mqxfa.b1l1': 'mqxfa13b',
    'mqxfa.a1l1': 'mqxfa18',
    'mqxfa.a1r1': 'mqxfa07b',
    'mqxfa.b1r1': 'mqxfa15',
    'mqxfb.a2r1': 'mqxfb03',
    'mqxfb.b2r1': 'mqxfb06',
    'mqxfa.a3r1': 'mqxfa14b',
    'mqxfa.b3r1': 'mqxfa08b',
    'mqxfa.b3l5': 'mqxfa22',
    'mqxfa.a3l5': 'mqxfa21',
    'mqxfb.b2l5': 'mqxfb04',
    'mqxfb.a2l5': 'mqxfb05',
    'mqxfa.b1l5': 'mqxfa11',
    'mqxfa.a1l5': 'mqxfa10',
    'mqxfa.a1r5': 'mqxfa19',
    'mqxfa.b1r5': 'mqxfa20',
    'mqxfb.a2r5': 'mqxfb07',
    'mqxfb.b2r5': 'mqxfb08',
    'mqxfa.a3r5': 'mqxfa23',
    'mqxfa.b3r5': 'mqxfa24',
}

rotated = {
    'mqxfa.b3l1': False,
    'mqxfa.a3l1': True,
    'mqxfb.b2l1': True,
    'mqxfb.a2l1': False,
    'mqxfa.b1l1': False,
    'mqxfa.a1l1': True,
    'mqxfa.a1r1': False,
    'mqxfa.b1r1': True,
    'mqxfb.a2r1': True,
    'mqxfb.b2r1': False,
    'mqxfa.a3r1': False,
    'mqxfa.b3r1': True,
    'mqxfa.b3l5': False,
    'mqxfa.a3l5': True,
    'mqxfb.b2l5': True,
    'mqxfb.a2l5': False,
    'mqxfa.b1l5': False,
    'mqxfa.a1l5': True,
    'mqxfa.a1r5': False,
    'mqxfa.b1r5': True,
    'mqxfb.a2r5': True,
    'mqxfb.b2r5': False,
    'mqxfa.a3r5': False,
    'mqxfa.b3r5': True,
}

data_files = {
'mqxfa03':  'FQ_MQXFA/MQXFA03_CA01.json',
'mqxfa04':  'FQ_MQXFA/MQXFA04_CA02.json',
'mqxfa05':  'FQ_MQXFA/MQXFA05_CA03.json',
'mqxfa06':  'FQ_MQXFA/MQXFA06_CA04.json',
'mqxfa07b': 'FQ_MQXFA/MQXFA07b_CA01.json',
'mqxfa08b': 'FQ_MQXFA/MQXFA08b_CA07.json',
'mqxfa10':  'FQ_MQXFA/MQXFA10_CA05.json',
'mqxfa11':  'FQ_MQXFA/MQXFA11_CA06.json',
'mqxfa12b': 'FQ_MQXFA/MQXFA12b_CA05.json',
'mqxfa13b': 'FQ_MQXFA/MQXFA13b_CA03.json',
'mqxfa14b': 'FQ_MQXFA/MQXFA14b_CA08.json',
'mqxfa15':  'FQ_MQXFA/MQXFA15_CA02.json',
'mqxfa17b': 'FQ_MQXFA/MQXFA17b_CA06.json',
'mqxfa18':  'FQ_MQXFA/MQXFA18_CA04.json',
'mqxfa19':  'FQ_MQXFA/MQXFA19_CA07.json',
'mqxfa20':  'FQ_MQXFA/MQXFA20_CA08.json',
'mqxfa21':  'FQ_MQXFA/MQXFA21_CA09.json',
'mqxfa22':  'FQ_MQXFA/MQXFA22_CA09.json',
'mqxfa23':  'FQ_MQXFA/MQXFA23_CA10.json',
'mqxfa24':  'FQ_MQXFA/MQXFA24_CA10.json',
'mqxfb03':  'FQ_MQXFB/FQ_MQXFB03_cold_nominal.json',
'mqxfb04':  'FQ_MQXFB/FQ_MQXFB04_cold_nominal.json',
'mqxfb05':  'FQ_MQXFB/FQ_MQXFB05_cold_nominal.json',
'mqxfb06':  'FQ_MQXFB/FQ_MQXFB06_cold_nominal.json',
'mqxfb07':  'FQ_MQXFB/FQ_MQXFB07_cold_nominal.json',
'mqxfb08':  'FQ_MQXFB/FQ_MQXFB08_cold_nominal.json',
'mqxfb09':  'FQ_MQXFB/FQ_MQXFB09_cold_nominal.json',
'mqxfb10':  'FQ_MQXFB/FQ_MQXFB10_cold_nominal.json',
'mqxfb11':  'FQ_MQXFB/FQ_MQXFB11_cold_nominal.json',
}


def convert_multipolar_expansion(magnet_meas_data, is_rotated, main_order, ref_radius):

    '''
    Convert multipolar expansion from magnet measurement convention to xsuite knl_rel and ksl_rel.

    Parameters
    ----------
    magnet_meas_data: dict
        Dictionary containing the multipolar expansion in the magnet measurement convention
        with keys like 'a1', 'b1', 'a2', 'b2', etc. Note that in this convention
        a1 is the dipole field, b1 is the quadrupole field, a2 is the sextupole field, etc.
    is_rotated: bool
        Whether the magnet is rotated with respect to the reference frame of the measurement. This affects the sign of the multipoles.
    main_order: int
        Reference order of the multipolar expansion (e.g. 1 for quadrupoles, 2 for sextupoles, etc.)
    ref_radius: float
        Reference radius in meters at which the multipolar expansion is defined.

    Returns
    -------
    knl_rel: np.ndarray
        Array of relative normal multipole coefficients in xsuite convention, indexed by order.
        Note that in this convention k0 is the dipole field, k1 is the quadrupole field, etc.
    ksl_rel: np.ndarray
        Array of relative skew multipole coefficients in xsuite convention, indexed by order.
        Note that in this convention k0s is the skew dipole field, k1s is the skew quadrupole field, etc.
    '''

    knl_rel = []
    ksl_rel = []
    for kk in magnet_meas_data:
        assert kk[0] in {'a', 'b'}, f"Unexpected key {kk} in magnet_meas_data, expected keys like 'a1', 'b1', etc."

        ii = int(kk[1:]) - 1 # Xsuite order (0 is dipole)

        if kk[0] == 'a':
            aa = magnet_meas_data[kk]
            bb = 0
        else:
            aa = 0
            bb = magnet_meas_data[kk]

        yrotfactor = -1 if is_rotated else 1

        # From magnet measurement convention to MADX convention
        dknlr_mad = 1e-4 * bb * (-1 * yrotfactor) ** (main_order + ii    )
        dkslr_mad = 1e-4 * aa * (-1 * yrotfactor) ** (main_order + ii + 1)

        # From MADX convention to knl
        kknn_rel = dknlr_mad * ref_radius**(main_order - (ii)) * factorial(ii) / factorial(main_order)
        kkss_rel = dkslr_mad * ref_radius**(main_order - (ii)) * factorial(ii) / factorial(main_order)

        # Extend the list if needed
        while len(knl_rel) <= ii:
            knl_rel.append(0)
            ksl_rel.append(0)

        if kk[0] == 'b':
            knl_rel[ii] = kknn_rel
        else:
            ksl_rel[ii] = kkss_rel

    return np.array(knl_rel), np.array(ksl_rel)

# ---------------------------

multipole_errors_b1 = {}
multipole_errors_b2 = {}
for nn in magnet_asset_association:
    asset_name = magnet_asset_association[nn]
    is_rotated = rotated[nn]

    data = xt.json.load(data_files[asset_name])

    magnet_meas_data = {}
    for mult in data['multipoles']:
        aaa = mult['an']
        bbb = mult['bn']
        nnn = mult['n']
        magnet_meas_data[f'a{nnn}'] = aaa
        magnet_meas_data[f'b{nnn}'] = bbb

    ref_radius = data['reference_radius_mm'] * 1e-3
    main_order = 1 # The are all quadrupoles

    knl_rel_b1, ksl_rel_b1 = convert_multipolar_expansion(
        magnet_meas_data=magnet_meas_data,
        is_rotated=is_rotated,
        main_order=main_order,
        ref_radius=ref_radius
    )

    knl_rel_b2, ksl_rel_b2 = convert_multipolar_expansion(
        magnet_meas_data=magnet_meas_data,
        is_rotated=not is_rotated,
        main_order=main_order,
        ref_radius=ref_radius
    )

    multipole_errors_b1[nn + '/lhcb1'] = {'knl_rel': knl_rel_b1, 'ksl_rel': ksl_rel_b1}
    multipole_errors_b2[nn + '/lhcb2'] = {'knl_rel': knl_rel_b2, 'ksl_rel': ksl_rel_b2}

    assert nn.startswith('mqx') # is a a normal quadrupole
    multipole_errors_b1[nn+'/lhcb1']['main_order'] = 1
    multipole_errors_b1[nn+'/lhcb1']['main_is_skew'] = False
    multipole_errors_b2[nn+'/lhcb2']['main_order'] = 1
    multipole_errors_b2[nn+'/lhcb2']['main_is_skew'] = False


env = xt.load('lhc_arc_errors.json')
set_multipole_errors_in_line(
    line=env['lhcb1'],
    multipole_errors=multipole_errors_b1,
    min_order=0, max_order=10,
    error_knob_name='on_error_triplets15',
    append_order_to_knob_name=True)

set_multipole_errors_in_line(
    line=env['lhcb2'],
    multipole_errors=multipole_errors_b2,
    min_order=0, max_order=10,
    error_knob_name='on_error_triplets15',
    append_order_to_knob_name=True)

env.to_json('lhc_arc_errors_arc_and_triplets15.json')