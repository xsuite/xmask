import xtrack as xt
import xmask.lhc as xmlhc
import xmask as xm

from pathlib import Path

test_data_dir = Path(__file__).parent.parent / "test_data"

def test_multipole_errors_and_correction():

    ###################
    # Load arc errors #
    ###################

    fname_rotations = (test_data_dir /
        "hllhc19/prepare_multipolar_errors/multipole_errors_pre_hllhc/magnet_orientation.tab")
    fname_err_table = (test_data_dir /
        "hllhc19/prepare_multipolar_errors/multipole_errors_pre_hllhc/collision_errors-emfqcs-6.tfs")

    min_order = 0
    max_order = 15

    multipole_errors_arcs, tt_err_arc = xmlhc.load_wise_table_arc_magnets(
        fname_err_table=fname_err_table,
        fname_rotations=fname_rotations,
        min_order=min_order, max_order=max_order)

    #############################
    # Load inner triplet errors #
    #############################

    multipole_errors_triplets15 = {}
    for nn in MAGNET_ASSET_ASSOCIATION_IT15:
        asset_name = MAGNET_ASSET_ASSOCIATION_IT15[nn]
        is_rot = IS_ROTATED[nn]

        magnet_meas_data = load_hllhc_multipole_json(test_data_dir /
            "hllhc19/prepare_multipolar_errors/multipole_errors_hllhc_ir15" /
            DATA_FILES_IT15[asset_name])
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

        multipole_errors_triplets15[nn + '/lhcb1'] = {'knl_rel': knl_rel_b1, 'ksl_rel': ksl_rel_b1}
        multipole_errors_triplets15[nn + '/lhcb2'] = {'knl_rel': knl_rel_b2, 'ksl_rel': ksl_rel_b2}

        order, is_skew = xmlhc.order_and_is_skew_from_name(nn)

        multipole_errors_triplets15[nn+'/lhcb1']['main_order'] = order
        multipole_errors_triplets15[nn+'/lhcb1']['main_is_skew'] = is_skew
        multipole_errors_triplets15[nn+'/lhcb2']['main_order'] = order
        multipole_errors_triplets15[nn+'/lhcb2']['main_is_skew'] = is_skew

        # Alternative naming convention
        multipole_errors_triplets15[nn + '/b1'] = multipole_errors_triplets15[nn + '/lhcb1']
        multipole_errors_triplets15[nn + '/b2'] = multipole_errors_triplets15[nn + '/lhcb2']

    ##################
    # Load D2 errors #
    ##################

    multipole_errors_d2_15 = {}

    for nn in MAGNET_ASSET_ASSOCIATION_D2_15:
        for beam in ['b1', 'b2']:
            nn_with_beam = nn + '.' + beam

            location = nn[-2:] # e.g "l5"
            aper = xmlhc.BEAM_MAPPING_PER_SIDE[location][beam] # e.g. 'v1' or 'v2'

            asset_name = MAGNET_ASSET_ASSOCIATION_D2_15[nn]
            is_rot = IS_ROTATED[nn]

            if is_rot: # Swap aper if rotated
                aper = 'v1' if aper == 'v2' else 'v2'

            data_file = DATA_FILES_D2_15[asset_name][aper]
            magnet_meas_data = load_hllhc_multipole_json(
                test_data_dir / "hllhc19/prepare_multipolar_errors/multipole_errors_hllhc_ir15" / data_file)
            ref_radius = magnet_meas_data.pop('ref_radius')

            assert nn.startswith('mbrd.') # is a a normal dipole
            main_order = 0
            main_is_skew = False

            knl_rel, ksl_rel = xmlhc.convert_multipolar_expansion(
                magnet_meas_data=magnet_meas_data,
                is_rotated=is_rot,
                main_order=main_order,
                ref_radius=ref_radius
            )

            multipole_errors_d2_15[nn_with_beam] = {'knl_rel': knl_rel, 'ksl_rel': ksl_rel,
                'main_order': main_order, 'main_is_skew': main_is_skew}

    #########################
    # Apply errors in lines #
    #########################

    # Association knob_name -> multipole errors
    multipole_errors_to_apply = {
        'on_error_arc': multipole_errors_arcs,
        'on_error_triplets_ir15': multipole_errors_triplets15,
        'on_error_d2_ir15': multipole_errors_d2_15
    }

    # Get a collider model
    env = xt.load(test_data_dir /
        "hllhc14_references_from_legacy/collider_errors_off_corrections_off.json")

    # Apply errors in lines
    min_order = 0
    max_order = 15

    for knob_name, multipole_errors in multipole_errors_to_apply.items():
        for line_name in ['lhcb1', 'lhcb2']:
            line = env[line_name]
            xm.set_multipole_errors_in_line(line, multipole_errors,
                                    min_order=min_order, max_order=max_order,
                                    error_knob_name=knob_name,
                                    append_order_to_knob_name=True)

    # Switch off errors of order 0 and 1
    env['on_error_arc_k0'] = 0
    env['on_error_arc_k0s'] = 0
    env['on_error_arc_k1'] = 0
    env['on_error_arc_k1s'] = 0
    env['on_error_triplets_ir15_k0'] = 0
    env['on_error_triplets_ir15_k0s'] = 0
    env['on_error_triplets_ir15_k1'] = 0
    env['on_error_triplets_ir15_k1s'] = 0
    env['on_error_d2_ir15_k0'] = 0
    env['on_error_d2_ir15_k0s'] = 0
    env['on_error_d2_ir15_k1'] = 0
    env['on_error_d2_ir15_k1s'] = 0

    #################################
    # Compute and apply corrections #
    #################################


    # Status of error knobs
    tt_err_knobs = env.vars.get_table().rows[r'on_error_.*']
    print("Error knobs in the environment:")
    tt_err_knobs.show()

    # Errors off to get reference twiss
    env.set(tt_err_knobs.name, 0)
    tw_b1 = env['lhcb1'].twiss4d(reverse=False) # Reference twiss
    tw_b2 = env['lhcb2'].twiss4d(reverse=False) # Reference twiss
    tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

    # errors back on
    for nn in tt_err_knobs.name:
        env[nn] = tt_err_knobs['value', nn]

    # Local correction of IR15 multipole errors
    xmlhc.correct_ir_errors(env, twiss_b1=tw_b1, twiss_b2=tw_b2,
                            corrections=IR_CORRECTIONS)

    # Spool piece correctors (MCS, MC0, MCD)
    xmlhc.set_arc_spool_piece_correctors(env, twiss_b1=tw_b1, twiss_b2=tw_b2)

    # k1s local + global correction (uses MQS)
    xmlhc.correct_k1s(env, twiss_b1=tw_b1, twiss_b2=tw_b2)

    # k2s local + global correction (uses MSS)
    xmlhc.correct_k2s(env, twiss_b1=tw_b1, twiss_b2=tw_b2)


# One possible configuration - final magnet-asset association still to be defined
MAGNET_ASSET_ASSOCIATION_IT15 = {
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

DATA_FILES_IT15 = {
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

IS_ROTATED = {
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
    'mbxf.4l1': False,
    'mbxf.4r1': True,
    'mbxf.4l5': False,
    'mbxf.4r5': True,

    'mbrd.4l1': True,
    'mbrd.4r1': False,
    'mbrd.4l5': True,
    'mbrd.4r5': False,

    # never used, to be checked
    # 'mcbrdh.4l1': False,
    # 'mcbrdv.4l1': True,
    # 'mcbrdv.4r1': False,
    # 'mcbrdh.4r1': True,
    # 'mcbrdh.4l5': False,
    # 'mcbrdv.4l5': True,
    # 'mcbrdv.4r5': False,
    # 'mcbrdh.4r5': True
}

MAGNET_ASSET_ASSOCIATION_D2_15 = {
    'mbrd.4l1': 'mbrd3',
    'mbrd.4r1': 'mbrd5',
    'mbrd.4l5': 'mbrd2',
    'mbrd.4r5': 'mbrd1'
}

DATA_FILES_D2_15 = {
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

IR_CORRECTIONS = {
  'ip1': {
    'range_b1': ['taxn.4l1/lhcb1', 'taxn.4r1/lhcb1'],
    'range_b2': ['taxn.4r1/lhcb2', 'taxn.4l1/lhcb2'],
    'corrections': {
      'on_corr_k2_ip1': {
        'correction_knobs': ['kcsx3.l1', 'kcsx3.r1'],
        'target_quantities_b1': {'f1020_b1': 'f1020'},
        'target_quantities_b2': {'f1020_b2': 'f1020'},
        'feed_down': False
      },
      'on_corr_k3_ip1': {
        'correction_knobs': ['kcox3.l1', 'kcox3.r1'],
        'target_quantities_b1': {'f4000_b1': 'f4000'},
        'target_quantities_b2': {'f4000_b2': 'f4000'},
        'feed_down': False
      },
      'on_corr_k4_ip1': {
        'correction_knobs': ['kcdx3.l1', 'kcdx3.r1'],
        'target_quantities_b1': {'f5000_b1': 'f5000'},
        'target_quantities_b2': {'f5000_b2': 'f5000'},
        'feed_down': False
      },
      'on_corr_k5_ip1': {
        'correction_knobs': ['kctx3.l1', 'kctx3.r1'],
        'target_quantities_b1': {'f6000_b1': 'f6000'},
        'target_quantities_b2': {'f6000_b2': 'f6000'},
        'feed_down': False
      },
      'on_corr_k1s_ip1': {
        'correction_knobs': ['kqsx3.l1', 'kqsx3.r1'],
        'target_quantities_b1': {'f1001_b1': 'f1001'},
        'target_quantities_b2': {'f1001_b2': 'f1001'},
        'feed_down': False
      },
      'on_corr_k2s_ip1': {
        'correction_knobs': ['kcssx3.l1', 'kcssx3.r1'],
        'target_quantities_b1': {'f0030_b1': 'f0030'},
        'target_quantities_b2': {'f0030_b2': 'f0030'},
        'feed_down': False
      },
      'on_corr_k3s_ip1': {
        'correction_knobs': ['kcosx3.l1', 'kcosx3.r1'],
        'target_quantities_b1': {'f1030_b1': 'f1030'},
        'target_quantities_b2': {'f1030_b2': 'f1030'},
        'feed_down': False
      },
      'on_corr_k4s_ip1': {
        'correction_knobs': ['kcdsx3.l1', 'kcdsx3.r1'],
        'target_quantities_b1': {'f0050_b1': 'f0050'},
        'target_quantities_b2': {'f0050_b2': 'f0050'},
        'feed_down': False
      },
      'on_corr_k5s_ip1': {
        'correction_knobs': ['kctsx3.l1', 'kctsx3.r1'],
        'target_quantities_b1': {'f1050_b1': 'f1050'},
        'target_quantities_b2': {'f1050_b2': 'f1050'},
        'feed_down': False
      }
    }
  },
  'ip5': {
    'range_b1': ['taxn.4l5/lhcb1', 'taxn.4r5/lhcb1'],
    'range_b2': ['taxn.4r5/lhcb2', 'taxn.4l5/lhcb2'],
    'corrections': {
      'on_corr_k2_ip5': {
        'correction_knobs': ['kcsx3.l5', 'kcsx3.r5'],
        'target_quantities_b1': {'f1020_b1': 'f1020'},
        'target_quantities_b2': {'f1020_b2': 'f1020'},
        'feed_down': False
      },
      'on_corr_k3_ip5': {
        'correction_knobs': ['kcox3.l5', 'kcox3.r5'],
        'target_quantities_b1': {'f4000_b1': 'f4000'},
        'target_quantities_b2': {'f4000_b2': 'f4000'},
        'feed_down': False
      },
      'on_corr_k4_ip5': {
        'correction_knobs': ['kcdx3.l5', 'kcdx3.r5'],
        'target_quantities_b1': {'f5000_b1': 'f5000'},
        'target_quantities_b2': {'f5000_b2': 'f5000'},
        'feed_down': False
      },
      'on_corr_k5_ip5': {
        'correction_knobs': ['kctx3.l5', 'kctx3.r5'],
        'target_quantities_b1': {'f6000_b1': 'f6000'},
        'target_quantities_b2': {'f6000_b2': 'f6000'},
        'feed_down': False
      },
      'on_corr_k1s_ip5': {
        'correction_knobs': ['kqsx3.l5', 'kqsx3.r5'],
        'target_quantities_b1': {'f1001_b1': 'f1001'},
        'target_quantities_b2': {'f1001_b2': 'f1001'},
        'feed_down': False
      },
      'on_corr_k2s_ip5': {
        'correction_knobs': ['kcssx3.l5', 'kcssx3.r5'],
        'target_quantities_b1': {'f0030_b1': 'f0030'},
        'target_quantities_b2': {'f0030_b2': 'f0030'},
        'feed_down': False
      },
      'on_corr_k3s_ip5': {
        'correction_knobs': ['kcosx3.l5', 'kcosx3.r5'],
        'target_quantities_b1': {'f1030_b1': 'f1030'},
        'target_quantities_b2': {'f1030_b2': 'f1030'},
        'feed_down': False
      },
      'on_corr_k4s_ip5': {
        'correction_knobs': ['kcdsx3.l5', 'kcdsx3.r5'],
        'target_quantities_b1': {'f0050_b1': 'f0050'},
        'target_quantities_b2': {'f0050_b2': 'f0050'},
        'feed_down': False
      },
      'on_corr_k5s_ip5': {
        'correction_knobs': ['kctsx3.l5', 'kctsx3.r5'],
        'target_quantities_b1': {'f1050_b1': 'f1050'},
        'target_quantities_b2': {'f1050_b2': 'f1050'},
        'feed_down': False
      }
    }
  },
}

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


