import xtrack as xt

def load_hllhc_multipole_json(fname):
    data = xt.json.load(fname)

    magnet_meas_data = {}
    for mult in data['multipoles']:
        aaa = mult['an']
        bbb = mult['bn']
        nnn = mult['n']
        magnet_meas_data[f'a{nnn}'] = aaa
        magnet_meas_data[f'b{nnn}'] = bbb

    magnet_meas_data['ref_radius'] = data['reference_radius_mm'] * 1e-3

    return magnet_meas_data


