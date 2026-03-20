import xtrack as xt
import xmask.lhc as xmlhc
from hllhc_ir15_geometry import is_rotated
from load_hl_multipole_json import load_hllhc_multipole_json

import numpy as np

# One possible configuration - final magnet-asset association still to be defined
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
    'mbxf.4l1': 'mbxf2',
    'mbxf.4r1': 'mbxf1',
    'mbxf.4l5': 'mbxf5',
    'mbxf.4r5': 'mbxf3',
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
'mbxf1': 'FQ_MBXF/FQ_MBXF1_cold_nominal.json',
'mbxf2': 'FQ_MBXF/FQ_MBXF2_cold_nominal.json',
'mbxf3': 'FQ_MBXF/FQ_MBXF3_cold_nominal.json',
'mbxf5': 'FQ_MBXF/FQ_MBXF5_cold_nominal.json',
}


# ---------------------------

multipole_errors = {}
for nn in magnet_asset_association:
    asset_name = magnet_asset_association[nn]
    is_rot = is_rotated[nn]

    magnet_meas_data = load_hllhc_multipole_json(data_files[asset_name])
    ref_radius = magnet_meas_data.pop('ref_radius')

    main_order = 1 # The are all quadrupoles

    knl_rel_b1, ksl_rel_b1 = xmlhc.convert_multipolar_expansion(
        magnet_meas_data=magnet_meas_data,
        is_rotated=is_rot,
        main_order=main_order,
        ref_radius=ref_radius
    )

    knl_rel_b2, ksl_rel_b2 = xmlhc.convert_multipolar_expansion(
        magnet_meas_data=magnet_meas_data,
        is_rotated=not is_rot,
        main_order=main_order,
        ref_radius=ref_radius
    )

    multipole_errors[nn + '/lhcb1'] = {'knl_rel': knl_rel_b1, 'ksl_rel': ksl_rel_b1}
    multipole_errors[nn + '/lhcb2'] = {'knl_rel': knl_rel_b2, 'ksl_rel': ksl_rel_b2}

    order, is_skew = xmlhc.order_and_is_skew_from_name(nn)

    multipole_errors[nn+'/lhcb1']['main_order'] = order
    multipole_errors[nn+'/lhcb1']['main_is_skew'] = is_skew
    multipole_errors[nn+'/lhcb2']['main_order'] = order
    multipole_errors[nn+'/lhcb2']['main_is_skew'] = is_skew

xt.json.dump(multipole_errors, 'multipole_errors_inner_triplet_d1_ir15.json')

