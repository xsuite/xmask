import xtrack as xt
import xmask.lhc as xmlhc
from hllhc_ir15_geometry import is_rotated
from load_hl_multipole_json import load_hllhc_multipole_json

magnet_asset_association = {
    'mbrd.4l1': 'mbrd3',
    'mbrd.4r1': 'mbrd5',
    'mbrd.4l5': 'mbrd2',
    'mbrd.4r5': 'mbrd1'
}

data_files = {
    'mbrd1': {'v1': 'FQ_MBRD/FQ_MBRD1_AP1_cold_nominal_extrapolation.json',
              'v2': 'FQ_MBRD/FQ_MBRD1_AP2_cold_nominal_extrapolation.json'},
    'mbrd2': {'v1': 'FQ_MBRD/FQ_MBRD2_AP1_cold_nominal_extrapolation.json',
              'v2': 'FQ_MBRD/FQ_MBRD2_AP2_cold_nominal_extrapolation.json'},
    'mbrd3': {'v1': 'FQ_MBRD/FQ_MBRD3_AP1_cold_nominal_extrapolation.json',
              'v2': 'FQ_MBRD/FQ_MBRD3_AP2_cold_nominal_extrapolation.json'},
    'mbrd4': {'v1': 'FQ_MBRD/FQ_MBRD4_AP1_cold_nominal_extrapolation.json',
              'v2': 'FQ_MBRD/FQ_MBRD4_AP2_cold_nominal_extrapolation.json'},
    'mbrd5': {'v1': 'FQ_MBRD/FQ_MBRD5_AP1_cold_nominal_extrapolation.json',
              'v2': 'FQ_MBRD/FQ_MBRD5_AP2_cold_nominal_extrapolation.json'}
}

multipole_errors = {}

for nn in magnet_asset_association:
    for beam in ['b1', 'b2']:
        nn_with_beam = nn + '.' + beam

        location = nn[-2:] # e.g "l5"
        aper = xmlhc.BEAM_MAPPING_PER_SIDE[location][beam] # e.g. 'v1' or 'v2'

        asset_name = magnet_asset_association[nn]
        is_rot = is_rotated[nn]

        if is_rot: # Swap aper if rotated
            aper = 'v1' if aper == 'v2' else 'v2'

        data_file = data_files[asset_name][aper]
        magnet_meas_data = load_hllhc_multipole_json(data_file)
        ref_radius = magnet_meas_data.pop('ref_radius')

        assert nn.startswith('mbrd.') # is a a normal dipole
        main_order = 0

        knl_rel, ksl_rel = xmlhc.convert_multipolar_expansion(
            magnet_meas_data=magnet_meas_data,
            is_rotated=is_rot,
            main_order=main_order,
            ref_radius=ref_radius
        )

        multipole_errors[nn_with_beam] = {'knl_rel': knl_rel, 'ksl_rel': ksl_rel}

xt.json.dump(multipole_errors, 'multipole_errors_inner_triplet_d2_ir15.json')
