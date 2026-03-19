import numpy as np
import xtrack as xt

kLMCSmax=0.471*2 /0.017**2*0.3/7000.
kLMCOmax=0.040*6 /0.017**3*0.3/7000.
kLMCDmax=0.100*24/0.017**4*0.3/7000.


DEFAULT_SPOOL_PIECE_CORRECTOR_LIMITS = {
    'kcs.a12b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a23b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a34b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a45b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a56b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a67b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a78b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a81b1':  (-kLMCSmax, kLMCSmax),
    'kcs.a12b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a23b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a34b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a45b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a56b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a67b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a78b2':  (-kLMCSmax, kLMCSmax),
    'kcs.a81b2':  (-kLMCSmax, kLMCSmax),
    'kco.a12b1':  (-kLMCOmax, kLMCOmax),
    'kco.a23b1':  (-kLMCOmax, kLMCOmax),
    'kco.a34b1':  (-kLMCOmax, kLMCOmax),
    'kco.a45b1':  (-kLMCOmax, kLMCOmax),
    'kco.a56b1':  (-kLMCOmax, kLMCOmax),
    'kco.a67b1':  (-kLMCOmax, kLMCOmax),
    'kco.a78b1':  (-kLMCOmax, kLMCOmax),
    'kco.a81b1':  (-kLMCOmax, kLMCOmax),
    'kco.a12b2':  (-kLMCOmax, kLMCOmax),
    'kco.a23b2':  (-kLMCOmax, kLMCOmax),
    'kco.a34b2':  (-kLMCOmax, kLMCOmax),
    'kco.a45b2':  (-kLMCOmax, kLMCOmax),
    'kco.a56b2':  (-kLMCOmax, kLMCOmax),
    'kco.a67b2':  (-kLMCOmax, kLMCOmax),
    'kco.a78b2':  (-kLMCOmax, kLMCOmax),
    'kco.a81b2':  (-kLMCOmax, kLMCOmax),
    'kcd.a12b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a23b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a34b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a45b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a56b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a67b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a78b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a81b1':  (-kLMCDmax, kLMCDmax),
    'kcd.a12b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a23b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a34b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a45b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a56b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a67b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a78b2':  (-kLMCDmax, kLMCDmax),
    'kcd.a81b2':  (-kLMCDmax, kLMCDmax),
}

def set_arc_spool_piece_correctors(env, twiss_b1, twiss_b2,
                corrector_limits=DEFAULT_SPOOL_PIECE_CORRECTOR_LIMITS):

    arc_multipoles_to_suppress = {
        'k2l': 'kcs',
        'k3l': 'kco',
        'k4l': 'kcd',
    }

    arcs = ['12', '23', '34', '45', '56', '67', '78', '81']
    beams = ['b1', 'b2']

    tw_b12 = {'b1': twiss_b1, 'b2': twiss_b2}

    for beam_name in beams:

        line = env[f'lhc{beam_name}']
        tw = tw_b12[beam_name]

        tt = line.get_table(attr=True)
        scale_multipole = np.zeros_like(tt.s)
        scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
        scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors

        for arc_name in arcs:
            # identify range for the arc
            if beam_name == 'b1':
                start = f's.ds.r{arc_name[0]}.{beam_name}'
                end = f'e.ds.l{arc_name[1]}.{beam_name}'
            else:
                start = f'e.ds.l{arc_name[1]}.{beam_name}'
                end = f's.ds.r{arc_name[0]}.{beam_name}'

            for multipole, knob_prefix in arc_multipoles_to_suppress.items():
                correction_knobs = [f'{knob_prefix}.a{arc_name}{beam_name}']
                target_quantities={'multipole_to_suppress': lambda tw, tt: tt[multipole]}

                vary = []
                for kk in correction_knobs:
                    vary.append(xt.Vary(kk, step=1e-5, limits=corrector_limits[kk]))

                # Usage:
                rdt_contrib = xt.IntegralOptimization(
                                        line=line,
                                        twiss=tw,
                                        start=start,
                                        end=end,
                                        vary=vary,
                                        target_quantities=target_quantities,
                                        generated_knob_name=f'on_corr_{knob_prefix}_arc{arc_name}_{beam_name}',
                                        scale_multipoles=scale_multipole)

                opt = rdt_contrib.correct()

                print("Before setting the knob:")
                line.vars.get_table().rows[correction_knobs].show()

                env[opt.knob_name] = 1.0
                print("After setting the knob:")
                line.vars.get_table().rows[correction_knobs].show()
